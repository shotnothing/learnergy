"""Employs SMTJ RBM + Fully-connected NN to solve the MNIST Classification problem

Attributes:
    config (Dictionary): Defines the config parameters to run the script with.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from learnergy.models.bernoulli.rbm import RBM
from learnergy.models.smtj import SMTJRBM
from data.mnist_smtj import SMTJMnistDataset

import numpy as np
import matplotlib.pyplot as plt

# Initialization
config = {}
loss_values = []
acc_values = []

## Overall
config['n_classes'] = 10
config['sigma_ratio'] = 0.5
config['sigma_initial_shift'] = 0
config['sigma_initial_slope'] = 0
config['dataset'] = SMTJMnistDataset
# config['dataset'] = torchvision.datasets.MNIST

## Step 1: SMTJ Sigmoid
config['layer_1'] = {}
config['layer_1']['layer_size'] = 100
config['layer_1']['batch_size'] = 100
config['layer_1']['epochs'] = 30
config['layer_1']['lr'] = 0.01

## Step 2: Linear
config['layer_2'] = {}
config['layer_2']['layer_size'] = config['n_classes'] # Naturally
config['layer_2']['batch_size'] = 100
config['layer_2']['epochs'] = 10
config['layer_2']['lr'] = 0.001

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

    layer1 = SMTJRBM(
        n_visible=784,
        n_hidden=config['layer_1']['layer_size'],
        steps=1,
        learning_rate=config['layer_1']['lr'],
        momentum=0,
        decay=0,
        temperature=1,
        use_gpu=True,
        sigma_ratio=config['sigma_ratio'],
        sigma_initial_shift=config['sigma_initial_shift'],
        sigma_initial_slope=config['sigma_initial_slope']
    )

    # Training an RBM
    layer1.fit(train, batch_size=config['layer_1']['batch_size'], epochs=config['layer_1']['epochs'])

    # Creating the Fully Connected layer to append on top of RBM
    layer2 = nn.Linear(layer1.n_hidden, config['n_classes'])

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
            x_batch = x_batch.reshape(x_batch.size(0), layer1.n_visible)

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

        # Calculate the test accuracy for the model:
        for x_batch, y_batch in tqdm(val_batch):
            # Flatenning the testing samples batch
            x_batch = x_batch.reshape(x_batch.size(0), layer1.n_visible)

            # Passing the batch down the model
            y = layer1(x_batch)

            # Calculating the fully-connected outputs
            y = layer2(y)

            # Calculating predictions
            _, preds = torch.max(y, 1)

            # Calculating validation set accuracy
            val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

        print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}")
        loss_values.append(train_loss / len(train_batch))
        acc_values.append(val_acc)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Sigmoid+Linear σ-shift: {}, σ-slope: {}".format(
        config['sigma_initial_shift'],
        config['sigma_initial_slope']
        ))

    ax1.plot(np.array(loss_values))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_ylim([0, 1])

    ax2.plot(np.array(acc_values))
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])

    plt.show()
