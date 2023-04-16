import torch.cuda
import torch.nn as nn
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader
from CNNModel import FashionMNISTModel
from Loader import FashionLoader
import matplotlib as plt
from torch.utils.tensorboard import SummaryWriter

from scripts.Model import Net


class FashionTrainer:
    # Placeholder params for later (model)
    def __init__(self, lr, epochs, crit=None, model_name=None):
        self.lr = lr
        self.epochs = epochs
        self.device = self.get_dev()

        self.writer = SummaryWriter()

        if model_name == "premade":
            self.model = Net()
        elif model_name == "cnn":
            self.model = FashionMNISTModel()
        elif model_name == "linear":
            #Add linear model here
            pass

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
        self.save()


    def validate(self):
            total_loss = 0.0
            total_correct = 0
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
                    for img, label in self.loader.training_loader:
                        inner_count += 1
                        # Send image and labels to device.
                        img = img.to(self.device)
                        label = label.to(self.device)
                        output = self.model(img)
                        loss = self.crit(output, label)
                        prediction = torch.argmax(self.model(img), dim=1)
                        correct = sum(prediction==label).item()
                        total_correct += correct

                        curr_loss = curr_loss + loss.item()




                        # Write to TB
                        self.writer.add_scalar('Training Loss', curr_loss / inner_count, global_step=epoch)
                        self.writer.add_scalar('Accuracy', correct / len(self.loader.training_loader), global_step=epoch)




                    # Update Total Loss
                    total_loss += curr_loss / (count + 1)
                    count += 1

                    # Print Statements
                    print("Epoch: ", epoch)
                    print("Total Loss: ", total_loss)
                    print("Current Loss: ", curr_loss)

            self.writer.close()
            self.save()

    def set_epochs(self, epochs):
        self.epochs = epochs

    def save(self):
        torch.save(self.model.state_dict(), "models/model.pt")

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
    trainer = FashionTrainer(lr=0.001, epochs=50, model_name="premade")
    trainer.train()
