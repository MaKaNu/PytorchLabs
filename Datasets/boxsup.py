""" This Module creates Dataset based on the BoxSupDataset Package
"""
from __future__ import absolute_import
from torchvision import transforms
from PIL import Image
import numpy as np
from absl import flags

from BoxSupDataset.nasa_box_sup_dataset import NasaBoxSupDataset
from BoxSupDataset.transforms.utils import ToTensor

from Datasets.utils.errors import LoadingError
import Datasets.utils.transforms as extented_transforms

NUM_CLASSES = 7
IGNORE_LABEL = 255

FLAGS = flags.FLAGS

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
    root_dir=FLAGS.dataset_path,
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

class BOXSUP(Dataset):
    """ Nasa Box Sup dataset. """

    def __init__(
        self, root_dir, classfile='classes_bxsp.txt', labeltype='mask' ,
        transform=None, target_transfrom=None):
        """
        Args:
            root_dir (string): Directory with img folder and label folder.
            transform (callable, optional): Optional transform to be applied.
        """
        assert (Path(root_dir) / 'Images').exists() and \
            (Path(root_dir) / 'Labels').exists(), \
            'rootDir has not the Correct Format or does not exists.'
        assert callable(transform), \
            'transform needs to be a callable.'
        assert labeltype in ('mask', 'image'), \
            'labeltype needs to be \'mask\' or \'image\''

        self.root_dir = Path(root_dir)
        self.labeltype = labeltype
        self.transform = transform
        self.target_transform = target_transfrom
        self.classes = pd.read_csv(Path(root_dir) / 'Labels' / Path(classfile))
        self.imgs = self.makeDataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        img_path, mask_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.labeltype == 'mask':
            mask = sio.loadmat(mask_path)['mask_data']
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = Image.open(mask_path)

        sample = {'image': img, 'label': mask}

        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform is not None:
            sample['label'] = self.target_transform(sample['label'])

        return sample

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'Attirbutes:'
                f'root_dir={self.root_dir},'
                f'transforms={self.transform},'
                f'classes={self.classes},'
                f'imgs={self.imgs}'
                )

    def makeDataset(self):
        """ Creates the items for the Dataset.
            Checks if label and the img are matching and returns the items list
            with img and label as tuple.
        """
        items = []
        img_path = self.root_dir / Path('Images')
        mask_path = self.root_dir / Path('Labels')
        if self.labeltype == 'image':
            label_files = sorted(glob.glob1(mask_path, "*.png"))
        elif self.labeltype == 'mask':
            label_files = sorted(glob.glob1(mask_path, "*.mat"))
        else:
            raise RuntimeError('{self.labeltype} is not defined!')
        image_files = sorted(glob.glob1(img_path, "*.png"))
        for index, img in enumerate(image_files):
            if label_files[index].split('_label')[0] == \
               img.split('.png')[0]:
                items.append((
                    self.root_dir / Path('Images') / Path(img),
                    self.root_dir / Path('Labels') / Path(label_files[index])
                    ))
            else:
                raise RuntimeError('img and label are not the same!')
        return items

    @property
    def root_dir(self):
        """ root_dir Getter"""
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value):
        if not isinstance(value, Path):
            raise TypeError("value needs to be of Type Path")
        self._root_dir = value

    @property
    def labeltype(self):
        """ labeltype Getter"""
        return self._labeltype

    @labeltype.setter
    def labeltype(self, value):
        if not isinstance(value, str):
            raise TypeError("value needs to be of Type str")
        self._labeltype = value

    @property
    def transform(self):
        """ transform Getter"""
        return self._transform

    @transform.setter
    def transform(self, value):
        if not (callable(value) or value is None):
            raise TypeError("value needs to be a callable")
        self._transform = value

    @property
    def target_transform(self):
        """ target_transform Getter"""
        return self._target_transform

    @target_transform.setter
    def target_transform(self, value):
        if not (callable(value) or value is None):
            raise TypeError("value needs to be a callable")
        self._target_transform = value

    @property
    def imgs(self):
        """ imgs Getter"""
        return self._imgs

    @imgs.setter
    def imgs(self, value):
        if not isinstance(value, list):
            raise TypeError("value needs to be of Type list")
        self._imgs = value

    @property
    def classes(self):
        """ classes Getter"""
        return self._classes

    @classes.setter
    def classes(self, value):
        if not isinstance(value, DataFrame):
            raise TypeError("value needs to be of Type DataFrame")
        self._classes = value
    