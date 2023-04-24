import torch.cuda
import torch.nn as nn
from torch import relu, max_pool2d

from torch.utils.data import Dataset, DataLoader


class FashionMNISTModel(nn.Module):
    def __init__(self):
        # Init parent const
        super(FashionMNISTModel, self).__init__()
        # Initialize 2 Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        # Initialize 2 Linear Layers
        self.fc1 = nn.Linear(in_features=(12 * 4 * 4), out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)

        self.out = nn.Linear(in_features=60, out_features=10)


    def forward(self, in_item):

        # do convolution 1
        in_item = self.conv1(in_item)
        in_item = relu(in_item)
        in_item = max_pool2d(in_item, kernel_size=2, stride=2)


        # Do convolution 2
        in_item = self.conv2(in_item)
        in_item = relu(in_item)
        in_item = max_pool2d(in_item, kernel_size=2, stride=2)

        # reshape for Linear
        in_item = torch.flatten(in_item, start_dim=1)
        
        # Do Linear 1
        in_item = self.fc1(in_item)
        in_item = relu(in_item)

        # Do Linear 2
        in_item = self.fc2(in_item)
        in_item = relu(in_item)

        in_item = self.out(in_item)
        return in_item
    






