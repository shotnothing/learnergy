"""Employs SMTJ sigmoid layer + Fully-connected NN to solve the MNIST Classification problem

Attributes:
    config (Dictionary): Defines the config parameters to run the script with.
"""
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

# Initialization
config = {}

## Overall
config['n_classes'] = 10
config['sigma_ratio'] = 0.5 # Not used, should it?
config['sigma_initial_shift'] = 1
config['sigma_initial_slope'] = 1
config['dataset'] = SMTJMnistDataset
# config['dataset'] = torchvision.datasets.MNIST

## Step 1: SMTJ Sigmoid
config['layer_1'] = {}
config['layer_1']['layer_size'] = 100
config['layer_1']['batch_size'] = 100 # Ignore for this script, both layers have same batches
config['layer_1']['epochs'] = 30
config['layer_1']['lr'] = 0.01

## Step 2: Linear NN
config['layer_2'] = {}
config['layer_2']['layer_size'] = config['n_classes'] # Naturally
config['layer_2']['batch_size'] = 100
config['layer_2']['epochs'] = 10
config['layer_2']['lr'] = 0.01

class SMTJSigmoid(nn.Module):

    """Not kept in it's own package because it is not related to RBM and is very niche. 
    Adapted from torch.nn.Linear.
    
    Attributes:
        bias (TYPE): Trainable bias.
        in_features (TYPE): Size of input.
        out_features (TYPE): Size of layer (i.e. output).
        shift (TYPE): Sigma shift.
        slope (TYPE): Sigma slope.
        weight (TYPE): Trainable weights.
    """
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def setup_slope_shift(self, slope: float, shift: float):
        """Initialize the random slope and shift.
        
        Args:
            slope (float): Sigma slope to be applied.
            shift (float): Sigma shift to be applied.
        """
        self.slope = nn.Parameter(torch.abs(torch.randn(config['layer_1']['layer_size']) * slope + 1), requires_grad=False)
        self.shift = nn.Parameter(torch.randn(config['layer_1']['layer_size']) * shift, requires_grad=False)

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        """Constructor for the SMTJ sigmoid layer.
        
        Args:
            in_features (int): Size of input.
            out_features (int): Size of layer (i.e. output).
            bias (bool, optional): Not used. Adapted from torch.nn.Linear.
            device (None, optional): Not used. Adapted from torch.nn.Linear.
            dtype (None, optional): Not used. Adapted from torch.nn.Linear.
        """
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
        self.setup_slope_shift(config['sigma_initial_slope'],config['sigma_initial_shift'])

    def reset_parameters(self) -> None:
        """To initialize the trainable parameters, taken straight from torch.nn.Linear
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Operation for forward pass.
        
        Args:
            x (torch.Tensor): Input data.
        
        Returns:
            torch.Tensor: Description
        """
        out = F.sigmoid(
            self.slope * (F.linear(x, self.weight))
            + self.slope * self.bias
            + self.shift
        )
        return out
        

if __name__ == '__main__':
    # Creating training and validation/testing dataset
    train = config['dataset'](
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test = config['dataset'](
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # Training SMTJ Sigmoid
    layer1 = SMTJSigmoid(784, config['layer_1']['layer_size'])

    # Creating the Fully Connected layer to append on top of RBM
    layer2 = nn.Linear(config['layer_1']['layer_size'], config['n_classes'])

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    optimizer = [
        optim.Adam(layer1.parameters(), lr=config['layer_2']['lr']),
        optim.Adam(layer2.parameters(), lr=config['layer_2']['lr']),
    ]

    # Creating training and validation batches
    train_batch = DataLoader(train, batch_size=config['layer_2']['batch_size'], shuffle=False, num_workers=1)
    val_batch = DataLoader(test, batch_size=config['layer_2']['batch_size'], shuffle=False, num_workers=1)

    # For amount of fine-tuning epochs
    for e in range(config['layer_2']['epochs']):
        print(f"Epoch {e+1}/{config['layer_2']['epochs']}")

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
            y = layer1(x_batch)

            # Calculating the fully-connected outputs
            y = layer2(y)

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
            y = layer1(x_batch)

            # Calculating the fully-connected outputs
            y = layer2(y)

            # Calculating predictions
            _, preds = torch.max(y, 1)

            # Calculating validation set accuracy
            val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

        print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}")