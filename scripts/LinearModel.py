import torch.cuda
import torch.nn as nn
from torch import relu, max_pool2d
import sys

from torch.utils.data import Dataset, DataLoader


class LinearFashionMNISTModel(nn.Module):
    def __init__(self):
        # Init parent const
        super(LinearFashionMNISTModel(), self).__init__()
        
        # update the recursion limit
        #sys.setrecursionlimit(5000)
        
        #I just know that it's 28 x 28 and the in features seem to be 784 for 28x28 images
        self.fc1 = nn.Linear(in_features=784, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.fc3 = nn.Linear(in_features=300, out_features=150)
        self.fc4 = nn.Linear(in_features=150, out_features=75)
        
        #10 out features, 1 output for each clothing item
        self.out = nn.Linear(in_features=75, out_features=10) 

        # Define sequential operation
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.ReLU(),
            self.fc4,
            nn.ReLU(),
            self.out,
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x
    






