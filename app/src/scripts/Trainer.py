import torch.cuda
import torch.nn as nn
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader
from Model import FashionMNISTModel
from Loader import FashionLoader


# from torch.utils.tensorboard import SummaryWriter # importing summarywriter for tensorboard
# writer = SummaryWriter()

class FashionTrainer:
    # Placeholder params for later (model)
    def __init__(self, lr, epochs, crit=None):
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

        model = FashionMNISTModel()
        total_loss = 0.0

        for epoch in range(self.epochs):

            model.train(True)

            curr_loss = 0.0

            count = 0
            for img, label in self.loader.training_loader:
                count += 1
                img = img.to(self.device)
                label = label.to(self.device)

                self.opt.zero_grad()

                output = model(img)

                loss = self.crit(output, label)
                loss.backward()

                self.opt.step()

                curr_loss += loss.item()

            total_loss = curr_loss / count

            print("Epoch: ", epoch)
            print("Total Loss: ", total_loss)
            print("Current Loss: ", curr_loss)

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


if __name__ == "__main__":
    trainer = FashionTrainer(lr=0.001, epochs=30)
    trainer.train()
