import numpy as np
import torch
from Model import Net
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

import Loader


class Model_Test:
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
            print(self.accuracy)

            report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)

            print(report)
            print("\n\n\n")

            self.print_results(report=report)


    def print_results(self, report):
        for x in range(len(self.loader.classes)):
            print("Accuracy for " + self.loader.classes[x] + "s" + " was", round((self.accuracy[x] * 100), 0), "%.")

        print("---F1-Scores---")

        for x in range(len(self.loader.classes)):
            print("The f1-score for", self.loader.classes[x] + "s", "class  was", report.get(str(x)).get('f1-score'), ".")



if __name__ == "__main__":
    # Placeholder, currently unable to train at home. Will train again tomorrow and run test script.
    run_test = Model_Test(model_path="models/model.pt", model_type="pre-trained")
    run_test.test()
