from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np

class SMTJMnistDataset(Dataset):

    def __init__(self, train):
        self.data = pd.read_csv("./data/mnist-train.csv")
        
        if train:
            self.data = self.data[0:5000]
        else:
            self.data = self.data[5000:6000]

        self.image = torch.tensor(np.array(self.data.drop("label", axis = 1) / 255, np.float32))
        self.target = torch.tensor(np.array(self.data["label"].values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.image[index], self.target[index]