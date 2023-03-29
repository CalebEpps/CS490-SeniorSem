import torch.cuda
import torch.nn as nn
from torch import relu, max_pool2d

from torch.utils.data import Dataset, DataLoader


class FashionMNISTModel(nn.Module):
    def __init__(self):
        # Init parent const
        super(FashionMNISTModel, self).__init__()

        self.cl_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.m_pool = nn.MaxPool2d(2, 2)
        self.cl_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.linear_1 = nn.Linear(in_features=(16 * 4 * 4), out_features=120)
        self.linear_2 = nn.Linear(in_features=120, out_features=80)

        self.out = nn.Linear(in_features=80, out_features=10)


    def forward(self, in_item):

        # do convolution 1
        in_item = self.cl_1(in_item)
        in_item = relu(in_item)
        # *|CURSOR_MARCADOR|*
        in_item = max_pool2d(in_item, kernel_size=2, stride=2)


        # Do convolution 2
        in_item = self.cl_2(in_item)
        in_item = relu(in_item)
        in_item = max_pool2d(in_item, kernel_size=2, stride=2)

        # reshape for Linear
        in_item = in_item.reshape(-1, 16 * 4 * 4)
        
        # Do Linear 1
        in_item = self.linear_1(in_item)
        in_item = relu(in_item)

        # Do Linear 2
        in_item = self.linear_2(in_item)
        in_item = relu(in_item)

        in_item = self.out(in_item)
        return in_item
    






