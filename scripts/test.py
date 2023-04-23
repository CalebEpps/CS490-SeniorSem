import torch
from Model import Net
from torchvision import transforms

import Loader


class Model_Test:
    def __init__(self, model_path, model_type):
        self.loader = Loader.FashionLoader(batch_size=128)
        self.data_set = self.loader.test_dataset
        self.model_path = model_path
        if model_type == 'pre-trained':
            self.model = Net()
            self.path = "placeholder"
        elif model_type == 'linear':
            # Change these
            self.model = Net()
        else:
            self.model = Net()

        self.classes = self.loader.classes
        # Initialize the lists for tracking predictions and correct count
        self.correct_predictions = {label: 0 for label in self.classes}
        self.total_predictions = {label: 0 for label in self.classes}


    def test(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        correct = 0
        total = len(self.data_set)

        with torch.no_grad():
            for img, label in self.loader.test_loader:
                output = self.model(img)
                _, prediction = torch.max(output, 1)

                for clss, out in zip(label, prediction):
                    # If correct, increment correct in label list
                    if clss == out:
                        self.correct_predictions[self.classes[clss]] += 1

                    self.total_predictions[self.classes[clss]] += 1

        # Console Print the predictions:

        for lbl, correct in self.correct_predictions.items():
            print("Class Name: " + lbl + " " + "Accuracy: ", 100 * float(correct) / self.total_predictions[lbl])

if __name__ == "__main__":
    # Placeholder, currently unable to train at home. Will train again tomorrow and run test script.
    run_test = Model_Test(model_path="models/model.pt", model_type="pre-trained")
    run_test.test()
