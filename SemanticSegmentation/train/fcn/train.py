""" This Module is for setting up the training on a specific dataset with fcn
    network.

    It uses sys.argv to specify the Dataset, the name of the results folder
    and probably more
"""
from __future__ import absolute_import
import sys
from pathlib import Path
from torch.backends import cudnn
from tensorboardX import SummaryWriter

from SemanticSegmentation.models.fcn8 import FCN8

cudnn.benchmark = True

CHKT_PATH = '../../ckpt'
EXP_NAME = 'Change this to sys.argv' # TODO
WRITER = SummaryWriter(Path(CHKT_PATH) / Path('exp' + EXP_NAME))

ARGS = {
    'epoch_num': 300,
    'lr': 1e-10,
    'weight_decay': 1e-4,
    'momentum': 0.95,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 20,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.1  # sample some validation results to display
}

def main(train_args, argv=[]):
    net = FCN8(num_classes=DATASET.num_classes).cuda()
