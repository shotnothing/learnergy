from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SMTJMnistDataset(Dataset):

    def __init__(self):
        self.data = pd.read_csv("./data/mnist-train.csv")

    def __getitem__(self, index):
        return self.data.drop("label", axis = 1), self.data["label"]