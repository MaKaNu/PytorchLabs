import unittest
from SemanticSegmentation.models.fcn8 import FCN8
from torch.autograd.gradcheck import gradcheck
from torch.autograd import Variable
import torch
import torch.nn as nn

class TestFCN8(unittest.TestCase):

    def setUp(self):
        self.model = FCN8(8, pretrained=True)
        torch.manual_seed(0)
        self.x = torch.rand((4, 3, 45, 45))

    def test_forward(self):
        self.assertEqual(self.model.forward(self.x).shape.numel(), 64800)
        self.assertEqual(str(self.model.forward(self.x).shape), 'torch.Size([4, 8, 45, 45])')
        print(self.model.named_parameters)

    def atest_sanity(self):
        input = (Variable(torch.randn(20, 20).double(), requires_grad=True), )
        model = nn.Linear(20, 1).double()
        test = gradcheck(model, input, eps=1e-6, atol=1e-4)
        print(test)

if __name__ == "__main__":
    unittest.main()
