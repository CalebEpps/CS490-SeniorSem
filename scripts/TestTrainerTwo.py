import torch.cuda
import torch.nn as nn
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader
#from Model import FashionMNISTModel
from LinearModel import FashionMNISTModel
from Loader import FashionLoader
import matplotlib as plt
from torch.utils.tensorboard import SummaryWriter

class FashionTrainer:
    # Placeholder params for later (model)
    def __init__(self, lr, epochs, crit=None):
        self.lr = lr
        self.epochs = epochs
        self.device = self.get_dev()

        self.writer = SummaryWriter()

        self.model = FashionMNISTModel()
        self.loader = FashionLoader()

        self.model = self.model.to(self.device)

        if crit is None:
            self.crit = nn.CrossEntropyLoss()
        else:
            self.crit = crit

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, validate=False):
        training_losses = []
        total_loss = 0.0
        # Begin Epoch Run
        for epoch in range(self.epochs):

            self.model.train(True)
            # reset current loss to 0 for next training iteration
            curr_loss = 0.0
            # iterator
            count = 0
            for img, label in self.loader.training_loader:
                # Send image and labels to device.
                img = img.to(self.device)
                label = label.to(self.device)
                # Set gradient
                self.opt.zero_grad()

                output = self.model(img)

                loss = self.crit(output, label)
                loss.backward()

                self.opt.step()
                # Calculate current loss
                curr_loss = loss.item()
                self.writer.add_scalar('Training Loss', curr_loss, global_step=epoch)
            # Update Total Loss
            total_loss += curr_loss / (count + 1)
            count += 1

            # Print Statements
            print("Epoch: ", epoch)
            print("Total Loss: ", total_loss)
            print("Current Loss: ", curr_loss)

        self.writer.close()



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
    trainer = FashionTrainer(lr=0.001, epochs=100)
    trainer.train()
