import torch.cuda
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from CNNModel import FashionMNISTModel
from LinearModel import LinearFashionMNISTModel
from LinearModel import SimpleLinearFashionMNISTModel
from Loader import FashionLoader
from Model import Net
#from LinearFashionMNISTModel import LinearFashionMNISTModel
import argparse
"""
Trains pytorch models on the Fashion MNIST dataset 

Args:
lr (float) Learning rate used by the optimizer
epochs (int) Number of epochs to train the model
crit (optional): loss function that is used If nothing is prpovided,then cross-entropy loss will be used
model_name (optional): The name of a pre-defined model Options are "premade", "CNN", "linear". If it isn't provided, 
a custom FashionMNISTModel will be used.

Attributes: 
self.lr(float) learning rate used by optimizer
self.epochs(int) number of epochs to train model 
self.device(self.get_dev) The device that will be acted upon
self.writer(SummaryWriter) logging purposes
self.model(NN Model) Pytorch model that will be trained
self.model_type(str) types of models used
self.loader(FashionLoader) used to load the dataset 
self.crit = (nn.CrossEntropyLoss) used for training the model
self.opt(Optimizer) updates parameters of the model

"""


class FashionTrainer:
    def __init__(self, lr, epochs, crit=None, model_name=None):
        self.lr = lr
        self.epochs = epochs
        self.device = self.get_dev()
        # Summary writer is used later for graphing and evaluation purposes
        self.writer = SummaryWriter()
        # Temporary condition statements that allow for model script to be selected
        if model_name == "premade":
            self.model = Net()
            self.model_type = "premade"
        elif model_name == "CNN":
            self.model = FashionMNISTModel()
            self.models_type = "CNN"
        elif model_name == "linear":
            self.model = LinearFashionMNISTModel()
            self.model_type = "linear"
            #sys.setrecursionlimit(5000)
        else:
            self.model = FashionMNISTModel()

        self.loader = FashionLoader()
        #self.loader = FashionLoader(batch_size=1)

        self.model = self.model.to(self.device)

        # Allows for setting of custom loss function
        if crit is None:
            self.crit = nn.CrossEntropyLoss()
        else:
            self.crit = crit

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        total_loss = 0.0
        # Begin Epoch Run
        for epoch in range(self.epochs):

            # Set the model to train mode
            self.model.train(True)

            # reset current loss to 0 for next training epoch
            curr_loss = 0.0
            correct = 0

            # iterators
            count = 0
            inner_count = 0
            for img, label in self.loader.training_loader:
                inner_count += 1
                # Send image and labels to device.
                img = img.to(self.device)
                label = label.to(self.device)

                self.opt.zero_grad()

                output = self.model(img)

                # Calculate the Loss each iteration
                loss = self.crit(output, label)
                loss.backward()

                self.opt.step()

                # Update Current Loss
                curr_loss = (curr_loss + loss.item())

                # Write to TB
                self.writer.add_scalar('Training Loss', curr_loss / inner_count, global_step=epoch)

            # Run Validation Method
            with torch.no_grad():
                current_accuracy = self.validate()
            # Write Accuracy
            self.writer.add_scalar('Accuracy', current_accuracy, global_step=epoch)

            # Update Total Loss
            total_loss += curr_loss / (count + 1)
            count += 1

            # Print Statements for debugging
            print("Epoch: ", epoch+1)
            print("Total Loss: ", total_loss / len(self.loader.training_loader))
            print("Current Loss: ", curr_loss / len(self.loader.training_loader))
            print("Accuracy: ", current_accuracy)

        self.writer.close()
        self.save()

    def validate(self):
        # Set model to eval mode
        self.model.eval()
        num_correct = 0
        total = 0
        # Use the dataset from the validation loader
        for img, label in self.loader.validation_loader:
            img = img.to(self.device)
            label = label.to(self.device)

            # Get the predicted output and set the number of correct
            output = torch.argmax(self.model(img), 1)
            num_output_correct = sum(output == label).item()

            # Add the current number of correct to the total.
            num_correct += num_output_correct
            total += len(img)
            # Returns the calculated accuracy
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

    parser = argparse.ArgumentParser(description="Model Training Script")

    parser.add_argument("model_to_train", type=str, help="Run the training script on specified model: premade CNN linear")
    parser.add_argument("learning_rate", type=float, help="Specify the learning rate for this training session.")
    parser.add_argument("epochs", type=int, help="Specify the number of epochs for this training session.")

    args = parser.parse_args()

    trainer = FashionTrainer(lr=args.learning_rate, epochs=args.epochs, model_name=args.model_to_train)

    #trainer = FashionTrainer(lr=0.001, epochs=100, model_name="premade")
    #trainer.train()
    #trainer = FashionTrainer(lr=0.001, epochs=100, model_name="linear")

    trainer.train()
