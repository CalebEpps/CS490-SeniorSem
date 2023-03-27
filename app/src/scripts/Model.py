import torch.cuda
import torch.nn as nn
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader


class FashionMNISTModel(nn.Module):
    def __init__(self):
        # Init parent const
        super(FashionMNISTModel, self).__init__()

        self.cl_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.linear_1 = nn.Linear(in_features=(26 * 26 * 32), out_features=128)
        self.linear_2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, in_item):

        # Converts frm 32x1x28x28 to 32x32x26x26
        in_item = self.cl_1(in_item)
        in_item = F.relu(input)

        # Converts 32 to 32x26x26
        in_item = in_item.flatten(1)

        # Previous to 32x128 
        in_item = self.linear_1(in_item)
        in_item = F.relu(in_item)

        # Previous to 32x10
        in_item = self.linear_2(in_item)
        output = F.softmax(in_item)

        return output
    






