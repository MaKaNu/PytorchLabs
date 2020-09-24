""" This Module is for setting up the training on a specific dataset with fcn
    network.

    It uses sys.argv to specify the Dataset, the name of the results folder
    and probably more
"""
from __future__ import absolute_import
from logging import log # TODO
from os import abort # TODO
import sys
import logging
from pathlib import Path
from importlib import import_module
import torch
from torch.backends import cudnn
from torch.utils import data

from SemanticSegmentation.utils.logger import CustomFormatter
from SemanticSegmentation.utils.hyperparams import SysArgWrapper, EnvParams, \
    TrainParams, LogLevel

from SemanticSegmentation.models.fcn8 import FCN8

cudnn.benchmark = True

def init_logger(wrpr):
    lvl = LogLevel(wrpr)
    print(lvl)

    # create logger with 'spam_application'
    logger = logging.getLogger("train.py")
    logger.setLevel(lvl.log_level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(lvl.log_level)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    if lvl.wrong_value:
        logger.error('Wrong Attribute for Logging level. uses DEBUG instead.')

    return logger

def main(argv):
    wrpr = SysArgWrapper(argv)
    logger = init_logger(wrpr)
    trainer = TrainParams(wrpr)
    env = EnvParams(wrpr)
    abort_training = False

    train_args = trainer()
    env_args = env()


    if env_args['dataset'] == 'test':
        logger.warning("Dataset Test is loaded. This Dataset is only a " +
            "Placeholder and the training will be aborted")
        abort_training = True

    print(env_args['dataset'])
    print(train_args)

    try:
        dataset = import_module('Datasets.' + env_args['dataset'])
        logger.info(repr(dataset))
    except ImportError as err:
        print('Error:', err)
        dataset = None

    if not abort_training and dataset is not None:
        net = FCN8(num_classes=dataset.NUM_CLASSES).cuda()

    # Check if Training should Continue at certain Snapshot and if load it.
        if len(train_args['snapshot'])==0:
            curr_epoch = 1
            train_args['best_record'] = {
                'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0,
                'mean_iu': 0, 'fwavacc': 0}
        else:
            print('training resumes from ' + train_args['snapshot'])
            ckpt_path, exp_name = env.checkpt_path, env.export_name
            net.load_state_dict(torch.load(ckpt_path / Path(exp_name) / Path(train_args['snapshot'])))
            split_snapshot = train_args['snapshot'].split('_')
            curr_epoch = int(split_snapshot[1]) + 1
            train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                        'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                        'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}
    
    # Set Network in Training Mode.
        net.train() 

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

if __name__ == '__main__':
    main(sys.argv)
