import torch.cuda
import torch.nn as nn
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader
from Model import FashionMNISTModel
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
        total_loss = 0.0
        # Begin Epoch Run
        for epoch in range(self.epochs):

            self.model.train(True)
            # reset current loss to 0 for next training iteration
            curr_loss = 0.0
            correct = 0
            # iterator
            count = 0
            inner_count = 0
            for img, label in self.loader.training_loader:
                inner_count += 1
                # Send image and labels to device.
                img = img.to(self.device)
                label = label.to(self.device)
                output = self.model(img)
                loss = self.crit(output, label)

                curr_loss = curr_loss + loss.item()

                # Write to TB
                self.writer.add_scalar('Training Loss', curr_loss / inner_count, global_step=epoch)

                # Set gradient
                self.opt.zero_grad()
                loss.backward()

                self.opt.step()
                # Calculate current loss

            # Update Total Loss
            total_loss += curr_loss / (count + 1)
            count += 1

            # Print Statements
            print("Epoch: ", epoch)
            print("Total Loss: ", total_loss)
            print("Current Loss: ", curr_loss)

        self.writer.close()

    def validate(self):
        total_loss = 0.0
        # Begin Epoch Run
        for epoch in range(self.epochs):

            self.model.eval()
            # reset current loss to 0 for next training iteration
            curr_loss = 0.0
            correct = 0
            # iterator
            count = 0
            inner_count = 0
            with torch.no_grad():
                for img, label in self.loader.validation_loader:
                    # Send image and labels to device.
                    inner_count += 1
                    img = img.to(self.device)
                    label = label.to(self.device)
                    output = self.model(img)
                    loss = self.crit(output, label)

                    curr_loss = curr_loss + loss.item()

                    correct += self.get_correct_preds(output, label)
                    # Write to TB
                    self.writer.filename_suffix = "Validate"
                    self.writer.add_scalar("Accuracy", correct / len(self.loader.training_set), epoch)
                    self.writer.add_scalar('Training Loss', curr_loss / inner_count, global_step=epoch)


            # Update Total Loss
            total_loss += curr_loss / (count + 1)
            count += 1

            # Print Statements
            print("Epoch: ", epoch)
            print("Total Loss: ", total_loss)
            print("Current Loss: ", curr_loss)
            print("Accuracy: ", (correct / len(self.loader.validation_loader)))
            print("\n")

        self.writer.close()
        self.save()

    def test(self):
        size = len(self.loader.validation_loader)




    def set_epochs(self, epochs):
        self.epochs = epochs

    def save(self):
        torch.save(self.model.state_dict(), "model.pt")

    def get_correct_preds(self, out, labels):
        return out.argmax(dim=1).eq(labels).sum().item()


    @staticmethod
    def has_gpu():
        return torch.cuda.is_available()

    def get_dev(self):
        if self.has_gpu():
            return torch.device("cuda:0")
        else:
            return "cpu"


if __name__ == "__main__":
    trainer = FashionTrainer(lr=0.001, epochs=50)
    trainer.validate()
