import torch.cuda
import torch.nn as nn
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader
from Model import FashionMNISTModel
from Loader import FashionLoader


class FashionTrainer:
    # Placeholder params for later (model)
    def __init__(self, lr, epochs, model=None, crit=None):
        self.lr = lr
        self.epochs = epochs
        self.device = self.get_dev()

        self.model = FashionMNISTModel()
        self.loader = FashionLoader()

        self.model = self.model.to(self.device)

        if crit is None:
            self.crit = nn.CrossEntropyLoss()
        else:
            self.crit = crit

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self):
        for epoch in range(self.epochs):
            loss = 0.0
            accuracy = 0.0

            trained_model = self.model.train()

            for x, (img, labels) in enumerate(self.loader.training_set):
                imgs = img.to(self.device)
                labels = labels.to(self.device)




    def set_epochs(self, epochs):
        self.epochs = epochs



    @staticmethod
    def has_gpu():
        return torch.cuda.is_available()

    def get_dev(self):
        if self.has_gpu():
            return torch.device("cuda:0")
        else:
            return "cpu"
