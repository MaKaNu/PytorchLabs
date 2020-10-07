""" This Module is for setting up the training on a specific dataset with fcn
    network.

    It uses sys.argv to specify the Dataset, the name of the results folder
    and probably more
"""
from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import random

from absl import app
from absl import flags
# from absl import logging

import sys
import logging
from pathlib import Path
from importlib import import_module
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from SemanticSegmentation.utils.logger import CustomFormatter as CF
from SemanticSegmentation.utils.hyperparams import SysArgWrapper, EnvParams, \
    TrainParams, LogLevel
from SemanticSegmentation.utils.misc import check_mkdir, evaluate, AverageMeter

from SemanticSegmentation.models.fcn8 import FCN8

cudnn.benchmark = True

FLAGS = flags.FLAGS

# FLAGS for Learning parameters
flags.DEFINE_integer('epoch_num', 300, 'Number of Epochs')
flags.DEFINE_float('learn_rate', 1e-10, 'Learning rate')
flags.DEFINE_float('weight_decay', 1e-4, 'weigth decay (L2 Penalty) for \
    Decouple Regularization.')
flags.DEFINE_float('momentum', 0.95, 'First coefficients used for computing \
    running averages of gradient and its square ')
flags.DEFINE_integer('lr_patience', 100,  'large patience denotes fixed \
    learn_rate') # TODO CHeck Helpinfo
flags.DEFINE_string('snapshot', '',  'empty string denotes learning from \
    scratch')
flags.DEFINE_integer('print_freq', 20, 'How often the Validation will be \
    printed') # TODO Check helpinfo
flags.DEFINE_bool('val_save_to_img_file', False, 'determine if val results \
    should be saved as img.')
flags.DEFINE_float('val_img_sample_rate', 0.1, 'validation to display rate')

# FLAGS for environment parameters
flags.DEFINE_string('checkpt_path', './SemanticSegmentation/ckpt', \
    'Checkpointpath for intermediate results.')
flags.DEFINE_string('export_name', 'fcn', 'Name of the training process')
flags.DEFINE_string('dataset', 'test', 'Trainingset which will be loaded.')
flags.DEFINE_bool('splitted', False, 'determine if the dataset needs to be \
    splitted or is already splitted in train valid and test.')
flags.DEFINE_integer('percentage', 80, 'Amount of training data (Example: \
    80% train 10% valid 10% test).')
flags.DEFINE_string('log_level', 'DEBUG', 'Defines the ')

def init_logger():
    """This Function initializes the prettified logger"""
    levels = {
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'NOTSET': logging.NOTSET
            }
    try:
        assert FLAGS.log_level in levels.keys()
    except AssertionError as assert_without_msg:
        raise AssertionError('Choose valid log level: %s'%list(levels.keys())) \
            from assert_without_msg

    log_level = levels[FLAGS.log_level]
    # create logger with 'spam_application'
    logger = logging.getLogger("train.py")
    logger.setLevel(log_level)

    # create console handler with a higher log level
    consoleh = logging.StreamHandler()
    consoleh.setLevel(log_level)

    consoleh.setFormatter(CF())

    logger.addHandler(consoleh)

    return logger

def init_trainparams():
    """ Initialize a dict based on gflags entries"""
    train_params = {
        'epoch_num': FLAGS.epoch_num,
        'lr': FLAGS.learn_rate,
        'weight_decay': FLAGS.weight_decay,
        'momentum': FLAGS.momentum,
        'lr_patience': FLAGS.lr_patience,
        'snapshot': FLAGS.snapshot,
        'print_freq': FLAGS.print_freq,
        'val_save_to_img_file': FLAGS.val_save_to_img_file,
        'val_img_sample_rate': FLAGS.val_img_sample_rate
    }
    return train_params

def main(argv):
    del argv # Unused from gflags

    writer = SummaryWriter(Path(FLAGS.checkpt_path) / \
                Path('exp') / Path(FLAGS.export_name))

    # wrpr = SysArgWrapper(argv)
    logger = init_logger()
    trainer = TrainParams(wrpr)
    env = EnvParams(wrpr)
    abort_training = False

    train_args = trainer()
    env_args = env()


    if env_args['dataset'] == 'test':
        logger.warning("Dataset Test is loaded. This Dataset is only a \
            Placeholder and the training will be aborted")
        abort_training = True

    logger.info('Loaded Dataset: %s', env_args['dataset'])
    for key, value in train_args.items():
        logger.info('Loaded Hyper Parameter: %s: %s', key, str(value))

    try:
        dataset = import_module('Datasets.' + env_args['dataset'])
        logger.debug(repr(dataset))
    except ImportError as err:
        print('Error:', err)
        dataset = None

    if not abort_training and dataset is not None:
        net = FCN8(num_classes=dataset.NUM_CLASSES).cuda()

    # Check if Training should Continue at certain Snapshot and if load it.
        ckpt_path, exp_name = env.checkpt_path, env.export_name
        if len(train_args['snapshot'])==0:
            curr_epoch = 1
            train_args['best_record'] = {
                'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0,
                'mean_iu': 0, 'fwavacc': 0}
        else:
            logger.info('training resumes from %s', train_args['snapshot'])
            net.load_state_dict(
                torch.load(
                    ckpt_path / Path(exp_name) / Path(train_args['snapshot'])
                    )
                )
            split_snapshot = train_args['snapshot'].split('_')
            curr_epoch = int(split_snapshot[1]) + 1
            train_args['best_record'] = {
                'epoch': int(split_snapshot[1]),
                'val_loss': float(split_snapshot[3]),
                'acc': float(split_snapshot[5]),
                'acc_cls': float(split_snapshot[7]),
                'mean_iu': float(split_snapshot[9]),
                'fwavacc': float(split_snapshot[11])}

    # Set Network in Training Mode.
        net.train()

        if env.splitted:
            count_image = len(dataset.Dataset)
            assert env.percentage in ['60', '70', '80']
            perc1 = int(env.percentage) / 100
            perc2 = (1 - perc1) / 2
            train_count = round(count_image * perc1)
            valid_count = round(count_image * perc2)
            test_count = count_image - train_count - valid_count
            assert count_image == (train_count + valid_count + test_count), \
                "Splitted Images doesn't match length of Dataset "

            dataset.Dataset.transform = dataset.input_transform
            dataset.Dataset.target_transform = dataset.target_transform

            trainset, validset, testset = torch.utils.data.random_split(
                dataset.Dataset, (train_count, valid_count, test_count)
            )

            logger.debug(str((len(trainset), len(validset), len(testset))))
        else:
            validset = None
            trainset = None
            logger.critical('NOT IMPLEMENTED!')

        train_loader = DataLoader(
            trainset, batch_size=1, num_workers=0, shuffle=True)
        val_loader = DataLoader(
            validset, batch_size=1, num_workers=0, shuffle=False)

        criterion = CrossEntropyLoss(
            reduction='sum', ignore_index=dataset.IGNORE_LABEL).cuda()

        optimizer = optim.Adam([
            {'params': [
        param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * train_args['learn_rate']},
            {'params': [
        param for name, param in net.named_parameters() if name[-4:] != 'bias'],
            'lr': train_args['learn_rate'],
            'weight_decay': train_args['weight_decay']}
        ], betas=(train_args['momentum'], 0.999))

        if len(train_args['snapshot']) > 0:
            optimizer.load_state_dict(torch.load(
                ckpt_path / Path(exp_name) / \
                Path('opt_' + train_args['snapshot'])))
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr']
            optimizer.param_groups[1]['lr'] = train_args['lr']

        check_mkdir(ckpt_path)
        check_mkdir(ckpt_path / Path(exp_name))
        actual_time = '_'.join(str(datetime.now()).split())
        actual_time = '_'.join(actual_time.split(':'))
        actual_time = '_'.join(actual_time.split('.'))
        # actual_time = 'file'
        ckpt_file = ckpt_path / Path(exp_name) / Path(actual_time + '.txt')
        ckpt_file.write_text(str(train_args) + '\n\n')

        scheduler = ReduceLROnPlateau(
            optimizer, 'min',
            patience=train_args['lr_patience'],
            min_lr=1e-10, verbose=True)
        for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
            train(
                train_loader, net, criterion, optimizer, epoch, train_args,
                env, dataset.NUM_CLASSES)
            val_loss = validate(
                val_loader, net, criterion, optimizer, epoch, train_args,
                dataset.restore_transform, dataset.visualize, env, logger)
            scheduler.step(val_loss)

def train(train_loader, net, criterion, optimizer,
          epoch, train_args, env, num_classes):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['image'], data['label']
        assert inputs.size()[2:] == labels.size()[1:3]
        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:3]
        assert outputs.size()[1] == num_classes

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data[0], N)

        curr_iter += 1
        env.writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg
            ))


def validate(
    val_loader, net, criterion, optimizer, epoch,
    train_args, restore, visualize, env, logger):

    net.eval()

    ckpt_path, exp_name = env.checkpt_path, env.export_name

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for _, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        inputs = Variable(inputs, volatile=True).cuda()
        gts = Variable(gts, volatile=True).cuda()

        outputs = net(inputs)
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data[0] / N, N)

        if random.random() > train_args['val_img_sample_rate']:
            inputs_all.append(None)
        else:
            inputs_all.append(inputs.data.squeeze_(0).cpu())
        gts_all.append(gts.data.squeeze_(0).cpu().numpy())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc = evaluate(
        predictions_all, gts_all, dataset.num_classes)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
        )
        torch.save(net.state_dict(), ckpt_path / Path(exp_name) / \
            Path(snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), ckpt_path / Path(exp_name) / \
            Path('opt_' + snapshot_name + '.pth'))

        to_save_dir = ckpt_path / Path(exp_name) /  Path(str(epoch))
        if train_args['val_save_to_img_file']:
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
            if data[0] is None:
                continue
            input_pil = restore(data[0])
            gt_pil = dataset.colorize_mask(data[1])
            predictions_pil = dataset.colorize_mask(data[2])
            if train_args['val_save_to_img_file']:
                input_pil.save(to_save_dir / Path('%d_input.png' % idx))
                predictions_pil.save(to_save_dir / Path('%d_prediction.png' % idx))
                gt_pil.save(to_save_dir / Path('%d_gt.png' % idx))
            val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                               visualize(predictions_pil.convert('RGB'))])
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        env.writer.add_image(snapshot_name, val_visual)

    logger.info(68 * '-')
    logger.info('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    logger.info('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    logger.info(68 * '-')

    env.writer.add_scalar('val_loss', val_loss.avg, epoch)
    env.writer.add_scalar('acc', acc, epoch)
    env.writer.add_scalar('acc_cls', acc_cls, epoch)
    env.writer.add_scalar('mean_iu', mean_iu, epoch)
    env.writer.add_scalar('fwavacc', fwavacc, epoch)
    env.writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    net.train()
    return val_loss.avg

if __name__ == '__main__':
    app.run(main)
    # main(sys.argv)
