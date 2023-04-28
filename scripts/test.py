import numpy as np
import torch
from Model import Net
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

import Loader


class Model_Test:
    """
    represents a model test instance
    
    Args:
    model_path(str): variable that stores the path to the model file
    model_type(str): variable that specifies the type of model
    
    Attributes:
    loader(Loader.FashionLoader): instance of FashionLoader class
    data_set: instance of test dataset 
    model: instance of pytorch model class net 
    classes (List[str]) : list of strings that contains names of the classes in datatset
    accuracy (float) stores accuracy of the model 
    report stores classification report of the model 
    printable_report stores classificationreport of the model in format that can be printed
    
    Variables: 
    self.path a placeholder variable
    
    
    Return: 
    NONE 
    """
    def __init__(self, model_path, model_type):
        # Initialize loader with batch size 128, (Batch size is irrelevant in this case)
        self.loader = Loader.FashionLoader(batch_size=128)
        self.data_set = self.loader.test_dataset
        # Set the model path. If time permits, add ability to use args
        self.model_path = model_path
        # Temporary use of condition statements to allow training of multiple models
        if model_type == 'pre-trained':
            self.model = Net()
            self.path = "placeholder"
        elif model_type == 'linear':
            # Change these
            self.model = Net()
        else:
            self.model = Net()

        self.classes = self.loader.classes
        self.accuracy = 0
        self.report = None
        self.printable_report = None
        
        """
        Runs the model on the test dataset and generates a classification report and matrix
        
        Args:
        None
        
        Attributes:
        y_pred(list) list to store predicted labels
        y_true(list)list to store true labels
        conf_matrix(array) store matrix
        self.accuracy(float) stores the accuracy of the model
        self.printable_report(str) classification report of the model that can be printed 
        self.report stores the report of the model
        
        Variables:
        self.model_path(string) 
        self.model_type(string)
        self.model: instance of pytorch model
        self.loader instance of dataloader
        img pytorch tensor that stores images
        label Pytorch sensor that stores a label
        output Pytorch sensor that stores output
        prediction pytorch sensor that stores predicted label
        mean_accuracy(float) stores the mean accuracy
        
        Returns:
        None 
        
        """

    def test(self):
        # Load the requested model and set to eval mode
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        y_pred = []
        y_true = []

        with torch.no_grad():
            for img, label in self.loader.test_loader:
                output = self.model(img)
                # Make Prediction
                _, prediction = torch.max(output, 1)
                y_pred.extend(prediction)
                # Extend Matrix with updated label (Needs to be numpy format, so convert)
                label = label.data.cpu().numpy()
                y_true.extend(label)

                conf_matrix = confusion_matrix(y_true, y_pred)

            # `conf_matrix.astype('float')` converts the confusion matrix to a float type.
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

            # According to docs, you can get the true accuracy by taking diagonal of confusion matrix
            self.accuracy = conf_matrix.diagonal()

            self.printable_report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=False)
            self.report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)

            # Output all results
            self.print_classification_report()
            self.print_results(report=self.report)
            self.show_accuracy()
            
        """
        This method prints the accuracy of the results, F1 scores, recall, and precision
        
        Args:
        report: classification report in dictionar format 
     
        Attributes:
        None
        
        Variables: 
        self instance of class
        X: integr representing the index of the current class being printed
        
           
        Returns: 
        None

        """

    def print_results(self, report):

        print("\n---Accuracy---")
        for x in range(len(self.loader.classes)):
            print("Accuracy for " + self.loader.classes[x] + "s" + " was", round((self.accuracy[x] * 100), 0), "%.")

        print("\n---F1-Scores---")
        for x in range(len(self.loader.classes)):
            print("The f1-score for", self.loader.classes[x] + "s", "class  was", report.get(str(x)).get('f1-score'),
                  ".")

        print("\n---Recall---")
        for x in range(len(self.loader.classes)):
            print("The recall for", self.loader.classes[x] + "s", "class  was", report.get(str(x)).get('recall'), ".")

        print("\n---Precision---")
        for x in range(len(self.loader.classes)):
            print("The precision for", self.loader.classes[x] + "s", "class  was", report.get(str(x)).get('precision'),
                  ".")
                  
                  """
                  Calculates and prints the total accuracy of the model of all the classes
                  
                  Args:
                  None
                  
                  Variables:
                  mean_accuracy(float): mean accuracy of all the classes
                    
                  Return:
                  None
                  """

    def show_accuracy(self):
        print("\n---Overall Accuracy---")
        mean_accuracy = 0
        for x in self.accuracy:
            mean_accuracy += x

        mean_accuracy /= len(self.accuracy)

        mean_accuracy *= 100

        print("The average accuracy across all classes was ", mean_accuracy, "%.")
        
        """
        Prints the classification report in a legible format
        
        Args: 
        None
        
        Returns:
        None
        
        """

    def print_classification_report(self):
        print("\n", self.printable_report)
        
        """
        script instantiates a model_test object, sets the model_path and model_type parameters, 
        and runs the test method to evaluate the pretrained model on the test dataset
        
        Args:
        None
        
        Attributes:
        None
        
        Variables:
        run_test: instance of model_test class
        model_path(string) stores the path to the pretrained model file 
        model_type(string) specifies the type of pretrained model to use
        
        """

if __name__ == "__main__":
    # Placeholder, currently unable to train at home. Will train again tomorrow and run test script.
    run_test = Model_Test(model_path="models/model.pt", model_type="pre-trained")
    run_test.test()
