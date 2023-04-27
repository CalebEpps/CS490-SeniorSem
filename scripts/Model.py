import torch
import torch.nn as nn
import torch.nn.functional as F

"""
An NNM Neural Network Model for imageclassification using the MNISTDATA SET

Attributes:
self.conv1 (nn.Conv2d) First convolutional layer with three input channels and 6 output
self.pool(nn.MaxPool2d) Max pooling with size of 2 by 2 
self.conv2(nnConv2d) second convolutional layer with 6 input channels and 16 output
self.fc1(nn.Linear) connected layer with 16x4x4 input features and 120 output
self.fc2(nn.Linear) connected layer with 120 input features and 84 output
self.fc3(nn.Linear) connected layer with 84 input features and 10 output

methods:
__init_(self) initializes NNM layers
forward(self,x) passses the NN

Return:
None

"""


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        """
        Passes the NN (Neural Network)
        
        Args:
        x(torch.flatten) input tensor shape 
        
        Returns: 
        Produces the shape of the tensor with batch size
        """

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
