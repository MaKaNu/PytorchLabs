""" This Module creates Dataset for Pascal VOC Dataset
"""

import numpy as np
from pathlib import Path
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

NUM_CLASSES = 21
IGNORE_LABEL = 255
ROOT = '/media/b3-542/LIBRARY/Datasets/VOC'

PALETTE = [             # color map
    0, 0, 0,            # 0=background
    128, 0, 0,          # 1=aeroplane
    0, 128, 0,          # 2=bicycle
    128, 128, 0,        # 3=bird
    0, 0, 128,          # 4=boat
    128, 0, 128,        # 5=bottle
    0, 128, 128,        # 6=bus
    128, 128, 128,      # 7=car
    64, 0, 0,           # 8=cat
    192, 0, 0,          # 9=chair
    64, 128, 0,         # 10=cow
    192, 128, 0,        # 11=diningtable
    64, 0, 128,         # 12=dog
    192, 0, 128,        # 13=horse
    64, 128, 128,       # 14=motorbike
    192, 128, 128,      # 15=person
    0, 64, 0,           # 16=potted plant
    128, 64, 0,         # 17=sheep
    0, 192, 0,          # 18=sofa
    128, 192, 0,        # 19=train
    0, 64, 128          # 20= tv/monitor
    ]

ZERO_PAD = 256 * 3 - len(PALETTE)
for i in range(ZERO_PAD):
    PALETTE.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(PALETTE)

    return new_mask

class VOC(data.Dataset):
    def __init__(self, root_dir, mode='train', joint_transform=None, sliding_crop=None,
                 transform=None, target_transform=None):
        self.root_dir = root_dir
        self.imgs = self.make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(Path(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train':
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)

    def make_dataset(self, mode):
        assert mode in ['train', 'val', 'test']
        items = []
        if mode == 'train':
            img_path = Path(self.root_dir, 'benchmark_RELEASE', 'dataset', 'img')
            mask_path = Path(self.root_dir, 'benchmark_RELEASE', 'dataset', 'cls')
            data_list = [l.strip('\n') for l in open(Path(
                self.root_dir, 'benchmark_RELEASE',
                'dataset', 'train.txt')).readlines()]
            for it in data_list:
                item = (
                    Path(img_path, it + '.jpg'),    # Path to train image
                    Path(mask_path, it + '.mat'))   # Path to train mask
                items.append(item)
        elif mode == 'val':
            img_path = Path(self.root_dir, 'VOCdevkit', 'VOC2012', 'JPEGImages')
            mask_path = Path(self.root_dir, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
            data_list = [l.strip('\n') for l in open(Path(
                self.root_dir, 'VOCdevkit', 'VOC2012', 'ImageSets',
                'Segmentation', 'seg11valid.txt')).readlines()]
            for it in data_list:
                item = (
                    Path(img_path, it + '.jpg'),    # Path to val image
                    Path(mask_path, it + '.png'))   # Path to val mask
                items.append(item)
        else:
            img_path = Path(self.root_dir, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
            data_list = [l.strip('\n') for l in open(Path(
                self.root_dir, 'VOCdevkit (test)', 'VOC2012', 'ImageSets',
                'Segmentation', 'test.txt')).readlines()]
            for it in data_list:
                items.append((img_path, it))
        return items
