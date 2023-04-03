import numpy as np
import pandas
import torch.cuda
import torchvision.utils
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

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.training_set = FashionMNIST(root="../../Dataset/data", train=True, download=True, transform=self.transform)
        self.training_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.validation_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=False)

    def display(self):
        img = self.training_set[0][0]
        label = self.training_set[0]
        plt.imshow(img.squeeze(), cmap="Greys")
        plt.show()
        print(label)

    def training_set_length(self):
        return len(self.training_set)

if __name__ == "__main__":
    categories = []
    fs = FashionLoader(batch_size=128)
    fs.display()

