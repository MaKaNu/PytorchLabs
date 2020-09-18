""" This Module is for setting up the training on a specific dataset with fcn
    network.

    It uses sys.argv to specify the Dataset, the name of the results folder
    and probably more
"""
from __future__ import absolute_import
import sys
import ast
from pathlib import Path
from collections import defaultdict
from torch.backends import cudnn
from tensorboardX import SummaryWriter

from SemanticSegmentation.models.fcn8 import FCN8

cudnn.benchmark = True

ADD_ARGS = None
CHKT_PATH = None
EXP_NAME = None
WRITER = None
ARGS = None



class SysArgWrapper():
    """ Handles all sys.argv exclusive the module. The method createValues is
    used to select standard or user values from a key.

    Attributes:
        arglist (list): List of arguments which are passed with sys.argv
        args (dict): Dict which was translated from arglist

    """

    def __init__(self, arglist:list) -> None:
        """ Removes first entry from arglist and creates based on this recuded
        list the dict args by cleaning the correct values in dict form.

        Args:
            arglist (list): argument list which should be sys.argv

        """
        self.arglist = arglist[1:]
        self.args = self.cleanArguments()

    @property
    def arglist(self):
        """ arglist getter """
        return self._arglist

    @arglist.setter
    def arglist(self, value):
        if not isinstance(value, list):
            raise TypeError("value needs to be of Type list")
        self._arglist = value

    @property
    def args(self):
        """ args Getter"""
        return self._args

    @args.setter
    def args(self, value):
        if not isinstance(value, dict):
            raise TypeError("value needs to be of Type dict")
        self._args = value

    def cleanArguments(self) -> dict:
        """ Creates a dict from self.arglist
            the accepted formats are:
                --key=value : sets the value  in the returned dict['key']
                -key value  : sets the value  in the returned dict['key']
                --key       : sets the value of returned dict['key'] to true
                -key        : sets the value of returned dict['key'] to true

            return:
                returns a dict with all set options.
        """
        ret_args = defaultdict(list)

        for index, k in enumerate(self.arglist):
            if index < len(self.arglist) - 1:
                key_a, val_b = k, self.arglist[index+1]
            else:
                key_a, val_b = k, None

            new_key = None
            val = None

            # double hyphen, equals
            if key_a.startswith('--') and '=' in key_a:
                new_key, val = key_a.split('=')

            # double hyphen, no equals
            # single hyphen, no arg
            elif (key_a.startswith('--') and '=' not in key_a) or \
                 (key_a.startswith('-') and (not val_b or \
                  val_b.startswith('-'))):
                val = True

            # single hypen, arg
            elif key_a.startswith('-') and val_b and not val_b.startswith('-'):
                val = val_b

            else:
                if (val_b is None) or (key_a == val):
                    continue
                raise ValueError(
                    'Unexpected argument pair: %s, %s' % (key_a, val_b))

            # santize the key
            key = (new_key or key_a).strip(' -')
            ret_args[key] = ast.literal_eval(val)

        return ret_args

    def createValues(self, keys:dict) -> None:
        """uses the class attribute args(dict) to set the values from self.args
        or use standard values for different params.
        """
        output = dict()
        for key , value in keys.items():
            if key in self.args.keys():
                output[key] = self.args[key]
                # setattr(self, key, self.args[key])
            else:
                output[key] = value
                # setattr(self, key, value)
        return output


class TrainParams():
    """ Class for reading the sys.argv and setting values or if not available
    use standard values.

    Attributes:
        arglist (list): List of arguments which are passed with sys.argv
        args (dict): Dict which was translated from arglist
        epoch_num (int): Number of epochs the training should run
        learn_rate (float): Learning rate of the optimizer
        weight_decay (float): Weight factor for preventing overgrowth
        momentum (float): Momentum Value for Optimizing with Momentum
        lr_patience (int): Number of Epochs to wait before learningrate change
        snapshot (str): String for Continuing a interrupted training
        print_freq (int): Number of epochs between printing training status
        val_save_to_img_file (bool): Enables to save best record imgs
        val_img_sample_rate (float): random rate for vaolidation
    """
    # pylint: disable=too-many-instance-attributes
    # 20 is reasonable in this case. (Or I dont know)

    def __init__(self, arglist):
        """ Constructor
        Args:
                arglist (list): sys.argv list
        """
        self.arglist = arglist[1:]
        self.args = self.cleanArguments()
        self.createValues()

    def __repr__(self) -> str:
        return 'TrainParams(arglist=%r), arguments:%r' \
            % (self.arglist, (self.epoch_num, self.learn_rate,
               self.weight_decay, self.momentum, self.lr_patience,
               self.snapshot, self.print_freq, str(self.val_save_to_img_file),
               self.val_img_sample_rate))

    def __call__(self) -> dict:
        return {
            'epoch_num': self.epoch_num,
            'learn_rate': self.learn_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'lr_patience': self.lr_patience,
            'snapshot': self.snapshot,
            'print_freq': self.print_freq,
            'val_save_to_img_file': self.val_save_to_img_file,
            'val_img_sample_rate': self.val_img_sample_rate
        }

    @property
    def arglist(self):
        """ arglist getter """
        return self._arglist

    @arglist.setter
    def arglist(self, value):
        if not isinstance(value, list):
            raise TypeError("value needs to be of Type list")
        self._arglist = value

    @property
    def args(self):
        """ args Getter"""
        return self._args

    @args.setter
    def args(self, value):
        if not isinstance(value, dict):
            raise TypeError("value needs to be of Type dict")
        self._args = value

    @property
    def epoch_num(self):
        """ epoch_num Getter"""
        return self._epoch_num

    @epoch_num.setter
    def epoch_num(self, value):
        if not isinstance(value, int):
            raise TypeError("value needs to be of Type int")
        self._epoch_num = value

    @property
    def learn_rate(self):
        """ learn_rate Getter"""
        return self._learn_rate

    @learn_rate.setter
    def learn_rate(self, value):
        if not isinstance(value, float):
            raise TypeError("value needs to be of Type float")
        self._learn_rate = value

    @property
    def weight_decay(self):
        """ weight_decay Getter"""
        return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, value):
        if not isinstance(value, float):
            raise TypeError("value needs to be of Type float")
        self._weight_decay = value

    @property
    def momentum(self):
        """ momentum Getter"""
        return self._momentum

    @momentum.setter
    def momentum(self, value):
        if not isinstance(value, float):
            raise TypeError("value needs to be of Type float")
        self._momentum = value

    @property
    def lr_patience(self):
        """ lr_patience Getter"""
        return self._lr_patience

    @lr_patience.setter
    def lr_patience(self, value):
        if not isinstance(value, int):
            raise TypeError("value needs to be of Type int")
        self._lr_patience = value

    @property
    def snapshot(self):
        """ snapshot Getter"""
        return self._snapshot

    @snapshot.setter
    def snapshot(self, value):
        if not isinstance(value, str):
            raise TypeError("value needs to be of Type str")
        self._snapshot = value

    @property
    def print_freq(self):
        """ print_freq Getter"""
        return self._print_freq

    @print_freq.setter
    def print_freq(self, value):
        if not isinstance(value, int):
            raise TypeError("value needs to be of Type int")
        self._print_freq = value

    @property
    def val_save_to_img_file(self):
        """ val_save_to_img_file Getter"""
        return self._val_save_to_img_file

    @val_save_to_img_file.setter
    def val_save_to_img_file(self, value):
        if not isinstance(value, bool):
            raise TypeError("value needs to be of Type bool")
        self._val_save_to_img_file = value

    @property
    def val_img_sample_rate(self):
        """ val_img_sample_rate Getter"""
        return self._val_img_sample_rate

    @val_img_sample_rate.setter
    def val_img_sample_rate(self, value):
        if not isinstance(value, float):
            raise TypeError("value needs to be of Type float")
        self._val_img_sample_rate = value

    def cleanArguments(self) -> dict:
        """ Creates a dict from self.arglist
            the accepted formats are:
                --key=value : sets the value  in the returned dict['key']
                -key value  : sets the value  in the returned dict['key']
                --key       : sets the value of returned dict['key'] to true
                -key        : sets the value of returned dict['key'] to true

            return:
                returns a dict with all set options.
        """
        ret_args = defaultdict(list)

        for index, k in enumerate(self.arglist):
            if index < len(self.arglist) - 1:
                key_a, val_b = k, self.arglist[index+1]
            else:
                key_a, val_b = k, None

            new_key = None
            val = None

            # double hyphen, equals
            if key_a.startswith('--') and '=' in key_a:
                new_key, val = key_a.split('=')

            # double hyphen, no equals
            # single hyphen, no arg
            elif (key_a.startswith('--') and '=' not in key_a) or \
                 (key_a.startswith('-') and (not val_b or \
                  val_b.startswith('-'))):
                val = True

            # single hypen, arg
            elif key_a.startswith('-') and val_b and not val_b.startswith('-'):
                val = val_b

            else:
                if (val_b is None) or (key_a == val):
                    continue
                raise ValueError(
                    'Unexpected argument pair: %s, %s' % (key_a, val_b))

            # santize the key
            key = (new_key or key_a).strip(' -')
            ret_args[key] = ast.literal_eval(val)

        return ret_args

    def createValues(self) -> None:
        """uses the class attribute args(dict) to set the values from self.args
        or use standard values for different params.
        """
        keys = {
            'epoch_num': 300,
            'learn_rate': 1e-10,
            'weight_decay': 1e-4,
            'momentum': 0.95,
            'lr_patience': 100,  # large patience denotes fixed learn_rate
            'snapshot': '',  # empty string denotes learning from scratch
            'print_freq': 20,
            'val_save_to_img_file': False,
            'val_img_sample_rate': 0.1  # validation to display rate
        }
        for key , value in keys.items():
            if key in self.args.keys():
                setattr(self, key, self.args[key])
            else:
                setattr(self, key, value)


class EnvParams():
    """ Description

    Attributes:
        checkpt_path (Path): Path for saving Checkpoints
        export_name (str): Export name for saving the modell
        writer (SummaryWriter): For Writing Summary? # TODO


    """

    def __init__(self, wrapper:SysArgWrapper) -> None:
        """ Constructor

        Args:
            wrapper (SysArgWrapper): Wrapper object which transforms sys.argv

        """
        keys = {  # Standard Values
            'checkpt_path': Path('../../ckpt').resolve(),
            'export_name': 'fcn',
            'writer': None,
            'dataset': (None, None)
        }
        # self.arglist = arglist[1:]
        # self.args = self.cleanArguments()
        param_dict = wrapper.createValues(keys)
        for key, value in param_dict.items():
            setattr(self, key, value)
        if self.writer is None:
            self.writer = SummaryWriter(self.checkpt_path / \
                Path('exp') / Path(self.export_name))

    def __repr__(self) -> str:
        return 'EnvParams(wrapper), arguments:%r' \
            % ((self.checkpt_path, self.export_name, self.writer, 
                self.dataset),)

    def __call__(self) -> dict:
        return {
            'checkpt_path': self.checkpt_path,
            'export_name': self.export_name,
            'writer': self.writer,
            'dataset': self.dataset
        }

    @property
    def checkpt_path(self):
        """ checkpt_path Getter"""
        return self._checkpt_path

    @checkpt_path.setter
    def checkpt_path(self, value):
        if not isinstance(value, Path):
            raise TypeError("value needs to be of Type str")
        self._checkpt_path = value

    @property
    def export_name(self):
        """ export_name Getter"""
        return self._export_name

    @export_name.setter
    def export_name(self, value):
        if not isinstance(value, str):
            raise TypeError("value needs to be of Type str")
        self._export_name = value

    @property
    def writer(self):
        """ writer Getter"""
        return self._writer

    @writer.setter
    def writer(self, value):
        if not (isinstance(value, SummaryWriter) or value is None):
            raise TypeError("value needs to be of Type SummaryWriter or None")
        self._writer = value

    @property
    def dataset(self):
        """ dataset Getter"""
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        if not (isinstance(value, tuple) and \
                all(isinstance(n, str) for n in value)):
            raise TypeError("value needs to be of Type tuple and items str")
        self._dataset = value


def main(argv):
    wrpr = SysArgWrapper(argv)
    trainer = TrainParams(wrpr)
    env = EnvParams(wrpr)

    train_args = trainer()
    env_args = env()
    print(env_args['dataset'][0], env_args['dataset'][1])

    try:
        dataset = getattr(import_module(
            'Datasets.' + env_args['dataset'][0]), env_args['dataset'][1])
        print(repr(dataset()))
    except ImportError as err:
        print('Error:', err)

    if not argv:
        net = FCN8(num_classes=DATASET.num_classes).cuda()


if __name__ == '__main__':
    main(sys.argv)
