import torchvision
from torch.utils.data import DataLoader

from recogners.datasets.opf import OPFDataset
from recogners.models.rbm import RBM

# Creating training dataset
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
#train = OPFDataset(path='data/boat.txt')

# Creating training batches
train_batches = DataLoader(train, batch_size=128, shuffle=True, num_workers=1)

# Creating an RBM
r = RBM(n_visible=784, n_hidden=128, learning_rate=0.1, steps=1, temperature=1)

# Training an RBM
r.fit(train_batches, epochs=100)