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
        
        """
       This is the training method that uses the provided parameters
       
       Args:
       img a batch of images that will be validated 
       label corresponding label for the images 

      Attributes:
      self.model(pytorch): the trained model used for validation
      self.device(pytorch): the device used for training the model
      self.loader.validation_loader(pytorch): the validation data loader containing validation images and labels
     
     Varibles: 
     total_loss(float): running sum of the loss calculated for all training examples seen so far in the current epoch
     cur_loss(float): he loss calculated for the current training example being processed
     correct(int): The number of training examples for which the predicted output matches the true output for the current epoch
     count(int): The number of training examples processed so far in the current epoch
     inner_count(int): The number of iterations within the current epoch. Used for calculating the current loss
     img(torch.Tensor): The image data of the current training example
     label(torch.Tensor) : The label (class) of the current training example
     output (torch.Tensor): The output of the model for the current training example
     loss(torch.Tensor): The loss calculated for the current training example
     current_accuracy(float): The accuracy of the model on the validation set after completing the current epoch of training 
      
      Return:
      NONE
        
        """

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
        """
        Calculates the accuracy of the trained model on the validation set
        
        Args:
        self: An instance of the FashionTrainer class
        
        Attributes:
        self.model (nn.Module): the PyTorch model used for validation
        self.device (torch.device): the device (CPU/GPU) used for training and validation
        self.loader.validation_loader (DataLoader): the validation data loader, used to iterate over the validation dataset
        
        Variables:
        num_correct (int): the number of samples that are correctly classified validatation
        total (int): total number of validation samples
        img (torch.Tensor): image tensor from the validation dataset
        label (torch.Tensor): the label tensor from the validation dataset
        output (torch.Tensor): the predicted output tensor from the PyTorch model
        num_output_correct (int): the number of correctly classified samples in the predicted output tensor
        
        
        Returns:
        float that calculates the accuracy of the model on the validation set
       
        """

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
    
    """
    Sets the number of epochs for training.

    Args:
        epochs (int): Number of epochs to train the model.

    Attributes:
        self.epochs (int): Number of epochs for training.

    Returns:
        None
    """

    def set_epochs(self, epochs):
        self.epochs = epochs
        
        """
        Save the trained PyTorch model to a file based on its type

    Args:
        None

    Attributes:
        self.model_type (str): A string representing the type of model, which is used to determine the filename.
        self.model (nn.Module): The PyTorch model to be saved.
    
    Returns:
        None
        """

    def save(self):
        if self.model_type == "premade":
            torch.save(self.model.state_dict(), "models/model.pt")
        elif self.model_type == "cnn":
            torch.save(self.model.state_dict(), "models/cnn_model.pt")
        elif self.model_type == "linear":
            torch.save(self.model.state_dict(), "models/linear_model.pt")
        else:
            print("Error Saving Model. Did you give the correct argument during intialization?")
    """
    Returns the number of correct predictions based on the predicted outputs and labels

        Args:
            out (torch.Tensor): The predicted output of the model
            labels (torch.Tensor): The true labels for the corresponding inputs
            
        Attributes:
            None.

        Returns:
            int: Number of correct predictions

        
    """

    def get_correct_preds(self, out, labels):
        return out.argmax(dim=1).eq(labels).sum().item()
    """
    Check if CUDA GPU is available.

Args: 
    None

Attributes:
    None
   

Returns:
    bool: True if CUDA GPU is available, False otherwise.
    """

    @staticmethod
    def has_gpu():
        return torch.cuda.is_available()
    """
    Return the PyTorch device for training and validation.
    
    Attributes:
            self.has_gpu(): A boolean indicating whether a GPU is available for use.
           
    Returns:
            torch.device or str: The PyTorch device for training and validation.
                If a GPU is available, returns torch.device("cuda:0"). Otherwise, returns "cpu".
        
    """

    def get_dev(self):
        if self.has_gpu():
            return torch.device("cuda:0")
        else:
            return "cpu"
"""
Parse command line arguments and start training a fashion classification model.

    Args:
        model_to_train (str): The name of the model to train. Must be one of "premade", "CNN", or "linear"
        learning_rate (float): The learning rate to use during training
        epochs (int): The number of epochs to train for


    Raises:
        ValueError: If an invalid model name is provided

    Attributes:
        trainer (FashionTrainer): An instance of the FashionTrainer class used to train the model
        
    
    Returns:
        None
    """

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
