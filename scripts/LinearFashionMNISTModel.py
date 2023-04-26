import torch.cuda
import torch.nn as nn
from torch import relu
import torch.nn.functional as F
import sys

from torch.utils.data import Dataset, DataLoader


class LinearFashionMNISTModel(nn.Module):
    def __init__(self):
        # Init parent const
        super().__init__()

        # I just know that it's 28 x 28 and the in features seem to be 784 for 28x28 images
        self.fc1 = nn.Linear(in_features=784, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=350)
        self.fc3 = nn.Linear(in_features=350, out_features=150)
        self.fc4 = nn.Linear(in_features=150, out_features=75)

        # 10 out features, 1 output for each clothing item
        self.out = nn.Linear(in_features=75, out_features=10)

    def forward(self, in_item):
        # flatten the input tensor
        in_item = torch.flatten(in_item)

        in_item = F.relu(self.fc1(in_item))
        in_item = F.relu(self.fc2(in_item))
        in_item = F.relu(self.fc3(in_item))
        in_item = F.relu(self.fc4(in_item))

        in_item = F.relu(self.out(in_item))

        return in_item



