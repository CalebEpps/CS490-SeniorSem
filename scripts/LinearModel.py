import torch.cuda
import torch.nn as nn
from torch import relu, max_pool2d

from torch.utils.data import Dataset, DataLoader


class FashionMNISTModel(nn.Module):
    def __init__(self):
        # Init parent const
        super(FashionMNISTModel, self).__init__()

        #this code wasn't working but I need some way to know what im starting with in the future
        #self.see = self.view(1, -1)
        #print(self.shape)
        
        #I just know that it's 28 x 28 and the in features seem to be 784 for 28x28 images
        # the in between I may test to see if i get better result from different I/Os
        #self.fc1 = nn.Linear(in_features=784, out_features=600)
        #got an error (mat1 and mat2 shapes cannot be multiplied (3584x28 and 784x600)
        #self.fl = torch.flatten(self, start_dim=0, end_dim=-1)
        self.fc1 = nn.Linear(in_features=784, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        self.fc3 = nn.Linear(in_features=300, out_features=150)
        self.fc4 = nn.Linear(in_features=150, out_features=75)
        #self.fc5 = nn.Linear(in_features=180, out_features=90)
        #self.fc6 = nn.Linear(in_features=90, out_features=60)
        
        #10 out features, 1 output for each clothing item
        self.out = nn.Linear(in_features=75, out_features=10) 


    def forward(self, in_item):
        #flatten the input tensor
        in_item = in_item.view(in_item.size(0), -1)
        
        in_item = relu(self.fc1(in_item))
        in_item = relu(self.fc2(in_item))
        in_item = relu(self.fc3(in_item))
        in_item = relu(self.fc4(in_item))
        #in_item = relu(self.fc5(in_item))
        #in_item = relu(self.fc6(in_item))
        
        in_item = relu(self.out(in_item))
        
        # do convolution 1
        #in_item = self.fc1(in_item)
        #in_item = relu(in_item)

        #in_item = max_pool2d(in_item, kernel_size=2, stride=2)


        # Do convolution 2
        #in_item = self.fc2(in_item)
        #in_item = relu(in_item)
        #in_item = max_pool2d(in_item, kernel_size=2, stride=2)

        # reshape for Linear
        #in_item = in_item.reshape(-1, 12 * 4 * 4)
        
        # Do Linear 1
        #in_item = self.linear_1(in_item)
        #in_item = relu(in_item)

        # Do Linear 2
        #in_item = self.linear_2(in_item)
        #in_item = relu(in_item)

        #in_item = self.out(in_item)
        return in_item
    






