""" This module includes the unittest class for the fcn8 model and a decorator
class for prittfy the console output."""

from __future__ import absolute_import
import unittest
import warnings

import torch
# from torch.nn import parameter
# from torch.autograd.gradcheck import gradcheck
# from torch.autograd import Variable
# import torch.nn as nn
from SemanticSegmentation.models.fcn8 import FCN8


class LineInfo():
    """ Deco Class for Lines """
    def __init__(self, lineformat:str = '-', linelength:int = 80,
                 info:tuple = ('', '')) -> None:
        self.lineformat = lineformat
        self.linelength = linelength
        self.info = info

    def __call__(self, func, *args, **kwargs):
        """ Prints Lines above and after the decorated func """
        def inner(*args, **kwargs):
            print(self.linelength * self.lineformat)
            print(self.info[0])
            func(*args, **kwargs)
            print(self.info[1])
            print(self.linelength * self.lineformat)
        return inner

    def __repr__(self) -> str:
        return 'LineInfo(lineformat=%r, linelength=%r, info=%r)' \
            % (self.lineformat, self.linelength, self.info)


class TestFCN8(unittest.TestCase):
    """ Unittest class for the fcn8 modell """
    def setUp(self):
        self.model = FCN8(8, pretrained=True)
        torch.manual_seed(0)
        # instead of zero init for score tensors use random init
        self.x = torch.rand((4, 3, 45, 45))

    @LineInfo(linelength=70, info=('Test Forward startet','Test Forward ended'))
    def testForward(self):
        """ Tests the FCN8.forward(x) function
        what is tested:
            - number of elements in output
            - shape of output
            - value at a specific index
            - max value if score is init zero_
        """

        self.changeWeights('random')
        output = self.model.forward(self.x)

        self.assertEqual(output.shape.numel(), 64800)
        self.assertEqual(list(output.shape), [4, 8, 45, 45])
        self.assertEqual(float(output[3][4][44][4]), 2362961408.0)

        self.changeWeights('zero')
        output = self.model.forward(self.x)

        self.assertEqual(torch.max(output), 0.0)

    def changeWeights(self, value:str = 'zero'):
        """ function to change the init value of score layers """
        if value == 'zero':
            self.model.score_fr[6].weight.data.zero_()
            self.model.score_fr[6].bias.data.zero_()
            self.model.score_pool3.weight.data.zero_()
            self.model.score_pool3.bias.data.zero_()
            self.model.score_pool4.weight.data.zero_()
            self.model.score_pool4.bias.data.zero_()
        elif value == 'random':
            self.model.score_fr[6].weight.data.random_()
            self.model.score_fr[6].bias.data.random_()
            self.model.score_pool3.weight.data.random_()
            self.model.score_pool3.bias.data.random_()
            self.model.score_pool4.weight.data.random_()
            self.model.score_pool4.bias.data.random_()
        else:
            warnings.warn(
                "weights stays unchanged use value == 'random' or 'zero'")



    '''def atest_sanity(self):
        input = (Variable(torch.randn(20, 20).double(), requires_grad=True), )
        model = nn.Linear(20, 1).double()
        test = gradcheck(model, input, eps=1e-6, atol=1e-4)
        print(test)
    '''



if __name__ == "__main__":
    unittest.main()
