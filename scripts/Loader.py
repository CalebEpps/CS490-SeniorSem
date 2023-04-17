import random

import numpy as np
import pandas
import torch.cuda
from torch.utils import data
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import logging


class FashionLoader(Dataset):
    ### Initialize dataset class, inherits parent
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        print(f"Batch Size set to {self.batch_size}")

        self.dataset = FashionMNIST(root="../../Dataset/data", train=True, download=True,
                                    transform=transforms.ToTensor())

        len_dataset = len(self.dataset)
        len_training_set = int(0.8 * len_dataset)
        len_validation_set = int(0.2 * len_dataset)

        print("dataset Length: ", len_dataset)
        print("Sum Lengths: ", len_validation_set, len_training_set)

        self.training_set, self.validation_set = data.random_split(self.dataset, (len_training_set, len_validation_set))

        # Divide training set into 2 pieces (80% training, 20% validation)
        self.training_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=False)
        self.validation_loader = DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False)

        self.classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    def training_set_length(self):
        return len(self.training_set)
