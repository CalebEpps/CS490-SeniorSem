from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class FashionLoader(Dataset):
    ### Initialize dataset class, inherits parent
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        print(f"Batch Size set to {self.batch_size}")
        # This is the full dataset
        self.dataset = FashionMNIST(root="../../Dataset/data", train=True, download=True,
                                    transform=transforms.Compose(
                                        [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]))
        # This is also the full dataset, but with training set to false
        self.test_dataset = FashionMNIST(root="../../Dataset/data", train=False, download=True,
                                         transform=transforms.Compose(
                                             [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]))
        # The validation set is 20% of the full dataset and the training set is 80%.
        len_dataset = len(self.dataset)
        len_training_set = int(0.8 * len_dataset)
        len_validation_set = int(0.2 * len_dataset)

        # Debugging Print Statements
        print("dataset Length: ", len_dataset)
        print("Sum Lengths: ", len_validation_set, len_training_set)
        # here the sets are randomly split. The random split method ensures no crossover
        self.training_set, self.validation_set = data.random_split(self.dataset, (len_training_set, len_validation_set))

        # These are the dataloaders used in training, validation, and testing
        self.training_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=False)
        self.validation_loader = DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # I put the classes here in the loader since it's the file that's intiailized
        # across the training and testing scripts.
        self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    # This is a simple debug method and is obsolete.
    def training_set_length(self):
        return len(self.training_set)
