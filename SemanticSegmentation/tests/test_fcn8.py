import unittest

# from torch.nn import parameter
# from torch.autograd.gradcheck import gradcheck
# from torch.autograd import Variable
import torch
# import torch.nn as nn
from SemanticSegmentation.models.fcn8 import FCN8


def lines(decorated):
    ''' Prints Lines above and after the decorated func'''
    def inner(*args, **kwargs):
        print(80 * '-')
        decorated(*args, **kwargs)
        print(80 * '-')
    return inner


class TestFCN8(unittest.TestCase):
    def setUp(self):
        self.model = FCN8(8, pretrained=True)
        torch.manual_seed(0)
        # instead of zero init for score tensors use random init
        self.model.score_fr[6].weight.data.random_()
        self.model.score_fr[6].bias.data.random_()
        self.model.score_pool3.weight.data.random_()
        self.model.score_pool3.bias.data.random_()
        self.model.score_pool4.weight.data.random_()
        self.model.score_pool4.bias.data.random_()
        self.x = torch.rand((4, 3, 45, 45))

    @lines
    def testForward(self):
        output = self.model.forward(self.x)

        self.assertEqual(output.shape.numel(), 64800)
        self.assertEqual(list(output.shape), [4, 8, 45, 45])
        self.assertEqual(float(output[3][4][44][4]), 2164337920.0)

    '''def atest_sanity(self):
        input = (Variable(torch.randn(20, 20).double(), requires_grad=True), )
        model = nn.Linear(20, 1).double()
        test = gradcheck(model, input, eps=1e-6, atol=1e-4)
        print(test)
    '''



if __name__ == "__main__":
    unittest.main()
