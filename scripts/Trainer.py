import torch.cuda
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from CNNModel import FashionMNISTModel
from Loader import FashionLoader
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
            self.model_type = "premade"
        elif model_name == "cnn":
            self.model = FashionMNISTModel()
            self.models_type = "cnn"
        elif model_name == "linear":
            self.model_type = "linear"
            # Add linear model here
        else:
            self.model = FashionMNISTModel()

        self.loader = FashionLoader()

        self.model = self.model.to(self.device)

        if crit is None:
            self.crit = nn.CrossEntropyLoss()
        else:
            self.crit = crit

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
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

                self.opt.zero_grad()

                output = self.model(img)

                loss = self.crit(output, label)
                loss.backward()

                self.opt.step()

                curr_loss = (curr_loss + loss.item())

                # Write to TB
                self.writer.add_scalar('Training Loss', curr_loss / inner_count, global_step=epoch)

            # Run Validation Method
            with torch.no_grad():
                current_accuracy = self.validate()
            # Write Accuracy
            self.writer.add_scalar('Accuracy', current_accuracy, global_step=epoch)

            # Set gradient

            # Calculate current loss

            # Update Total Loss
            total_loss += curr_loss / (count + 1)
            count += 1

            # Print Statements
            print("Epoch: ", epoch)
            print("Total Loss: ", total_loss / len(self.loader.training_loader))
            print("Current Loss: ", curr_loss / len(self.loader.training_loader))
            print("Accuracy: ", current_accuracy)

        self.writer.close()
        self.save()

    def validate(self):
        self.model.eval()
        num_correct = 0
        total = 0

        for img, label in self.loader.validation_loader:
            img = img.to(self.device)
            label = label.to(self.device)

            output = torch.argmax(self.model(img), 1)
            num_output_correct = sum(output == label).item()

            num_correct += num_output_correct
            total += len(img)

            return num_correct / total

    def set_epochs(self, epochs):
        self.epochs = epochs

    def save(self):
        if self.model_type == "premade":
            torch.save(self.model.state_dict(), "models/model.pt")
        elif self.model_type == "cnn":
            torch.save(self.model.state_dict(), "models/cnn_model.pt")
        elif self.model_type == "linear":
            torch.save(self.model.state_dict(), "models/linear_model.pt")
        else:
            print("Error Saving Model. Did you give the correct argument during intialization?")

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
    trainer = FashionTrainer(lr=0.001, epochs=100, model_name="premade")
    trainer.train()
