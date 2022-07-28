"""SMTJ-based RBM."""
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class SMTJMnistDataset(Dataset):
    """Data loader class that is a drop-in replacement for torchvision.datasets.MNIST, 
    for loading of data from a local CSV file.
    
    Attributes:
        data (Dataframe): Pandas dataframe of the read CSV file, used for easy
                          data manipulation.
        image (Tensor): Data of the pixels in each image, normalized.
        target (Tensor): Target label of the corresponding image.
    """
    
    def __init__(self, train, root="./data", download=None, transform=None):
        """Constructor for the SMTJMnistDataset class.
        
        Args:
            train (Boolean): Flag to determine if data should be first 5000 rows
                             (training dataset) or next 1000 rows (test dataset)
            root (String, optional): Directory pointing to the CSV file to load from.
            download (None, optional): Unused, left in for compatibility with torchvision.
            transform (None, optional): Unused, left in for compatibility with torchvision.
        """
        self.data = pd.read_csv(root + "/mnist-train.csv")
        
        if train:
            self.data = self.data[0:5000]
        else:
            self.data = self.data[5000:6000]

        self.image = torch.tensor(np.array(self.data.drop("label", axis = 1) / 255, np.float32))
        self.target = torch.tensor(np.array(self.data["label"].values))

    def __len__(self):
        """Overrides the special method for the len() built-in method.
        
        Returns:
            Integer: Number of rows of the data.
        """
        return len(self.data)

    def __getitem__(self, index):
        """Overrides the special method that determiens the return from data[index] calls.
        
        Args:
            index (Integer): Index of data to access
        
        Returns:
            Tupule: Image targer pairs at selected index.
        """
        return self.image[index], self.target[index]