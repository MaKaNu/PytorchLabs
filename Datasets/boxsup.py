""" This Module creates Dataset based on the BoxSupDataset Package
"""
from __future__ import absolute_import
from BoxSupDataset.nasa_box_sup_dataset import NasaBoxSupDataset
from numpy.core.fromnumeric import mean
from torchvision import transforms
from PIL import Image
import numpy as np

from Datasets.utils.errors import LoadingError
import Datasets.utils.transforms as extented_transforms

NUM_CLASSES = 7
IGNORE_LABEL = 255
ROOT = 'D:/Mitarbeiter/Kaupenjohann/09_GIT/PyTorch_Nasa_Dataset/data/TestBatch'


# Try to load the Dataset
try:
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])

    target_transform = extented_transforms.MaskToTensor()
    restore_transform = transforms.Compose([
        extented_transforms.DeNormalize(*mean_std),
        transforms.ToPILImage(),
    ])
    visualize = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(400),
        transforms.ToTensor()
    ])

    Dataset = NasaBoxSupDataset(
    classfile='classes_bxsp.txt',
    root_dir=ROOT,
    transform=transforms.Compose(
        [transforms.ToTensor(),
        ]
    ))
except AssertionError as err:
    raise LoadingError(err.args[0].split(' ',1)[0]) from err

# color map
# 0=background, 1=sand, 2=soil, 3=bedrock, 4=big rocks, 5=sky, 6=robot

palette = [0, 0, 0]

for obj_cls in Dataset.classes.iterrows():
    palette.extend(list(obj_cls[1])[1:])

ZERO_PAD = 256 * 3 - len(palette)
for i in range(ZERO_PAD):
    palette.append(0)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """ Colorize a given mask based on the palette values.
    Args:
        mask: numpy array of the mask
    Reutrn:
        new_mask: numpy array of the colored mask
    """
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
