import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from learnergy.models.bernoulli.rbm import RBM
from learnergy.models.smtj import SMTJRBM
from data.mnist_smtj import SMTJMnistDataset

# Defining some input variables

# Global
n_classes = 10
sigma_ratio=0.5
sigma_initial_shift=100
sigma_initial_slope=100

# Step 1: RBM only
batch_size = 100
epochs = 30
lr = 0.01

# Step 2: RBM + NN
batch_size_batch_size = 100
fine_tune_epochs = 20
fine_tune_lr = 0.01

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


    model = SMTJRBM(
        n_visible=784,
        n_hidden=100,
        steps=1,
        learning_rate=lr,
        momentum=0,
        decay=0,
        temperature=1,
        use_gpu=True,
        sigma_ratio=sigma_ratio,
        sigma_initial_shift=sigma_initial_shift,
        sigma_initial_slope=sigma_initial_slope
    )

    # Training an RBM
    model.fit(train, batch_size=batch_size, epochs=epochs)

    # Creating the Fully Connected layer to append on top of RBM
    fc = nn.Linear(model.n_hidden, n_classes)

    # Check if model uses GPU
    if model.device == "cuda":
        # If yes, put fully-connected on GPU
        fc = fc.cuda()

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    optimizer = [
        optim.Adam(model.parameters(), lr=fine_tune_lr),
        optim.Adam(fc.parameters(), lr=fine_tune_lr),
    ]

    # Creating training and validation batches
    train_batch = DataLoader(train, batch_size=batch_size_batch_size, shuffle=False, num_workers=1)
    val_batch = DataLoader(test, batch_size=batch_size_batch_size, shuffle=False, num_workers=1)

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
            x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if model.device == "cuda":
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

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

        # Calculate the test accuracy for the model:
        for x_batch, y_batch in tqdm(val_batch):
            # Flatenning the testing samples batch
            x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if model.device == "cuda":
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # Passing the batch down the model
            y = model(x_batch)

            # Calculating the fully-connected outputs
            y = fc(y)

            # Calculating predictions
            _, preds = torch.max(y, 1)

            # Calculating validation set accuracy
            val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

        print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}")

    # Saving the fine-tuned model
    torch.save(model, "tuned_model.pth")

    # Checking the model's history
    print(model.history)
