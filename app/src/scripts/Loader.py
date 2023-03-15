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
    def __init__(self, batch_size=64, logging_level="info"):
        # Logging starts for Loader File
        logging.basicConfig(filename="logs/loader_logs.txt", encoding='utf-8', level=self.set_logging_level(log_level=logging_level))

        self.batch_size = batch_size
        logging.info(('Batch Size set to ', self.batch_size))

        self.transform = transforms.Compose([transforms.ToTensor()])
        logging.info("Transform Composition Complete")

        logging.info("Training Set Initializing...")
        self.training_set = FashionMNIST(root="../../Dataset/data", train=True, download=True, transform=self.transform)
        logging.info("Training Set Ready")

        logging.info("Setting up Data Loader")
        self.training_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        logging.info("Data Loader Setup Complete")

    def display(self):
        img, label = next(iter(self.training_set))
        plt.imshow(img.squeeze(), cmap="gray")
        print(label)

    def training_set_length(self):
        return len(self.training_set)

    def set_logging_level(self, log_level):
        if log_level == "debug":
            logging.basicConfig(level=logging.DEBUG)
        elif log_level == "info":
            logging.basicConfig(level=logging.INFO)
        elif log_level == "warning":
            logging.basicConfig(level=logging.WARNING)
        else:
            logging.basicConfig(level=logging.ERROR)


if __name__ == "__main__":
    categories = []
    fs = FashionLoader(batch_size=64)
    fs.display()

