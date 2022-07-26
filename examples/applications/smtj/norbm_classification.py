import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from learnergy.models.bernoulli.rbm import RBM
from learnergy.models.smtj import SMTJRBM
from data.mnist_smtj import SMTJMnistDataset

# Defining some input variables
hist = {}

# Global
n_classes = 10
sigma_ratio=0.5
sigma_initial_shift=1
sigma_initial_slope=1

# Step 1: SMTJ Sigmoid
size = 100
batch_size = 100
epochs = 30
lr = 0.01

# Step 2: Linear
fine_tune_batch_size = 100
fine_tune_epochs = 10
fine_tune_lr = 0.001

class SMTJSigmoid(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def setup_slope_shift(self, slope: float, shift: float):
        self.slope = nn.Parameter(torch.abs(torch.randn(size) * slope + 1), requires_grad=False)
        self.shift = nn.Parameter(torch.randn(size) * shift, requires_grad=False)

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SMTJSigmoid, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.setup_slope_shift(sigma_initial_slope,sigma_initial_shift)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.sigmoid(
            self.slope * (F.linear(x, self.weight))
            + self.slope * self.bias
            + self.shift
        )
        return out
        

if __name__ == '__main__':
    # Creating training and validation/testing dataset
    train = SMTJMnistDataset(train=True)
    test = SMTJMnistDataset(train=False)

    # train = torchvision.datasets.MNIST(
    #     root="./data",
    #     train=True,
    #     download=True,
    #     transform=torchvision.transforms.ToTensor(),
    # )
    # test = torchvision.datasets.MNIST(
    #     root="./data",
    #     train=False,
    #     download=True,
    #     transform=torchvision.transforms.ToTensor(),
    # )

    # Training SMTJ Sigmoid
    model = SMTJSigmoid(784, size)

    # Creating the Fully Connected layer to append on top of RBM
    fc = nn.Linear(size, n_classes)

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    optimizer = [
        optim.Adam(model.parameters(), lr=fine_tune_lr),
        optim.Adam(fc.parameters(), lr=fine_tune_lr),
    ]

    # Creating training and validation batches
    train_batch = DataLoader(train, batch_size=fine_tune_batch_size, shuffle=False, num_workers=1)
    val_batch = DataLoader(test, batch_size=fine_tune_batch_size, shuffle=False, num_workers=1)

    # For amount of fine-tuning epochs
    for e in range(fine_tune_epochs):
        print(f"Epoch {e+1}/{fine_tune_epochs}")

        # Resetting metrics
        train_loss, val_acc = 0, 0

        # For every possible batch
        for x_batch, y_batch in tqdm(train_batch):
            # For every possible optimizer
            for opt in optimizer:
                # Resets the optimizer
                opt.zero_grad()

            # Flatenning the samples batch
            x_batch = x_batch.reshape(x_batch.size(0), 784)

            # Passing the batch down the model
            y = model(x_batch)

            # Calculating the fully-connected outputs
            y = fc(y)

            # Calculating loss
            loss = criterion(y, y_batch)

            # Propagating the loss to calculate the gradients
            loss.backward()

            # For every possible optimizer
            for opt in optimizer:
                # Performs the gradient update
                opt.step()

            # Adding current batch loss
            train_loss += loss.item()

        # Calculate the validation accuracy for the model:
        for x_batch, y_batch in tqdm(val_batch):
            # Flatenning the testing samples batch
            x_batch = x_batch.reshape(x_batch.size(0), 784)

            # Passing the batch down the model
            y = model(x_batch)

            # Calculating the fully-connected outputs
            y = fc(y)

            # Calculating predictions
            _, preds = torch.max(y, 1)

            # Calculating validation set accuracy
            val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

        print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}")