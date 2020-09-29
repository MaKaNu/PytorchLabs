""" This Module includes all Dataclasses used for PytrochLab"""
from __future__ import absolute_import
import ast
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from tensorboardX import SummaryWriter


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
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f' ATTRIBUTES:'
                f' arglist={self.arglist},' 
                f' args={self.args},'
                )

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


@dataclass
class TrainParams():
    """ Class for reading the sys.argv and setting values or if not available
    use standard values.

    Attributes:
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
    wrapper: SysArgWrapper

    def __post_init__(self):
        keys = {  # Standard Values
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
        param_dict = self.wrapper.createValues(keys)
        for key, value in param_dict.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f' ATTRIBUTES:'
                f' epoch_num={self.epoch_num},' 
                f' learn_rate={self.learn_rate},'
                f' weight_decay={self.weight_decay},'
                f' momentum={self.momentum},'
                f' lr_patience={self.lr_patience},'
                f' snapshot={self.snapshot},'
                f' print_freq={self.print_freq},' 
                f' val_save_to_img_file={self.val_save_to_img_file},'
                f' val_img_sample_rate={self.val_img_sample_rate}'
                )

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
    def wrapper(self):
        """ wrapper Getter"""
        return self._wrapper

    @wrapper.setter
    def wrapper(self, value):
        if not isinstance(value, SysArgWrapper):
            raise TypeError("value needs to be of Type SysArgWrapper")
        self._wrapper = value

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


@dataclass
class EnvParams():
    """ Description

    Attributes:
        checkpt_path (Path): Path for saving Checkpoints
        export_name (str): Export name for saving the modell
        writer (SummaryWriter): For Writing Summary? # TODO


    """
    wrapper: SysArgWrapper

    def __post_init__(self) -> None:
        keys = {  # Standard Values
            'checkpt_path': Path('./SemanticSegmentation/ckpt').resolve(),
            'export_name': 'fcn',
            'writer': None,
            'dataset': 'test',
            'splitted': False,
            'percentage': '80'
        }
        param_dict = self.wrapper.createValues(keys)
        for key, value in param_dict.items():
            setattr(self, key, value)
        if self.writer is None:
            self.writer = SummaryWriter(self.checkpt_path / \
                Path('exp') / Path(self.export_name))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f' ATTRIBUTES:'
                f' checkpt_path={self.checkpt_path},' 
                f' export_name={self.export_name},'
                f' writer={self.writer},' 
                f' dataset={self.dataset},'
                f' splitted={self.splitted},'
                f' percentage={self.percentage}'
                )

    def __call__(self) -> dict:
        return {
            'checkpt_path': self.checkpt_path,
            'export_name': self.export_name,
            'writer': self.writer,
            'dataset': self.dataset,
            'splitted': self.splitted,
            'percentage': self.percentage
        }

    @property
    def wrapper(self):
        """ wrapper Getter"""
        return self._wrapper

    @wrapper.setter
    def wrapper(self, value):
        if not isinstance(value, SysArgWrapper):
            raise TypeError("value needs to be of Type SysArgWrapper")
        self._wrapper = value

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
        if not (isinstance(value, str)):
            raise TypeError("value needs to be of Type str")
        self._dataset = value

    @property
    def splitted(self):
        """ splitted Getter"""
        return self._splitted
    
    @splitted.setter
    def splitted(self, value):
        if not isinstance(value, bool):
            raise TypeError("value needs to be of Type bool")
        self._splitted = value

    @property
    def percentage(self):
        """ percentage Getter"""
        return self._percentage
    
    @percentage.setter
    def percentage(self, value):
        if not isinstance(value, str):
            raise TypeError("value needs to be of Type str")
        self._percentage = value


@dataclass
class LogLevel():
    """ This Class holds the LogLevel, which could be set with sys.argv

    Attributes:
        Level (logging.LEVEL): Debug Level
        wrong_value (bool): True, if the transmitted value of sys.argv is not 
            accepted

    """
    wrapper: SysArgWrapper
    
    def __post_init__(self) -> None:
        keys = {  # Standard Values
            'log_level': 'DEBUG',
        }
        param_dict = self.wrapper.createValues(keys)
        for key, value in param_dict.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f' ATTRIBUTES:'
                f' log_level={self.log_level},' 
                f' wrong_value={self.wrong_value}')

    def __call__(self) -> dict:
        return {
            'log_level': self.log_level,
        }

    @property
    def wrapper(self):
        """ wrapper Getter"""
        return self._wrapper

    @wrapper.setter
    def wrapper(self, value):
        if not isinstance(value, SysArgWrapper):
            raise TypeError("value needs to be of Type SysArgWrapper")
        self._wrapper = value

    @property
    def wrong_value(self):
        """ wrong_value Getter"""
        return self._wrong_value
    
    @wrong_value.setter
    def wrong_value(self, value):
        if not isinstance(value, bool):
            raise TypeError("value needs to be of Type bool")
        self._wrong_value = value

    @property
    def log_level(self):
        """ log_level Getter"""
        return self._log_level
    
    @log_level.setter
    def log_level(self, value):
        if not isinstance(value, str):
            raise TypeError("value needs to be of Type str")

        levels = {
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'NOTSET': logging.NOTSET
            }

        if value in levels.keys():
            self._log_level = levels[value]
            self.wrong_value = False
        else:
            self._log_level = levels['DEBUG']
            self.wrong_value = True
