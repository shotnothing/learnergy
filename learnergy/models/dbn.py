import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torchvision import transforms

import learnergy.utils.constants as c
import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.core.dataset import Dataset
from learnergy.core.model import Model
from learnergy.models.dropout_rbm import DropoutRBM
from learnergy.models.e_dropout_rbm import EDropoutRBM
from learnergy.models.gaussian_rbm import GaussianRBM, VarianceGaussianRBM
from learnergy.models.rbm import RBM
from learnergy.models.sigmoid_rbm import SigmoidRBM

logger = l.get_logger(__name__)

MODELS = {
    'bernoulli': RBM,
    'dropout': DropoutRBM,
    'e_dropout': EDropoutRBM,
    'gaussian': GaussianRBM,
    'sigmoid': SigmoidRBM,
    'variance_gaussian': VarianceGaussianRBM
}


class DBN(Model):
    """A DBN class provides the basic implementation for Deep Belief Networks.

    References:
        G. Hinton, S. Osindero, Y. Teh. A fast learning algorithm for deep belief nets. Neural computation (2006).

    """

    def __init__(self, model='bernoulli', n_visible=128, n_hidden=[128], steps=1, learning_rate=0.1, momentum=0, decay=0, temperature=1, use_gpu=False):
        """Initialization method.

        Args:
            model (str): Indicates which type of RBM should be used to compose the DBN.
            n_visible (int): Amount of visible units.
            n_hidden (list): Amount of hidden units per layer.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: Model -> DBN.')

        # Override its parent class
        super(DBN, self).__init__(use_gpu=use_gpu)

        # Amount of visible units
        self.n_visible = n_visible

        # Amount of hidden units per layer
        self.n_hidden = n_hidden

        # Number of layers
        self.n_layers = len(n_hidden)

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        # Temperature factor
        self.T = temperature

        # List of models (RBMs)
        self.models = []

        # For every possible layer
        for i in range(self.n_layers):
            # If it is the first layer
            if i == 0:
                # Gathers the number of input units as number of visible units
                n_input = self.n_visible

            # If it is not the first layer
            else:
                # Gathers the number of input units as previous number of hidden units
                n_input = self.n_hidden[i-1]

            # Creates an RBM
            m = MODELS[model](n_input, self.n_hidden[i], self.steps,
                              self.lr, self.momentum, self.decay, self.T, use_gpu)

            # Appends the model to the list
            self.models.append(m)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')
        logger.debug(f'Number of layers: {self.n_layers}.')

    @property
    def n_visible(self):
        """int: Number of visible units.

        """

        return self._n_visible

    @n_visible.setter
    def n_visible(self, n_visible):
        if not isinstance(n_visible, int):
            raise e.TypeError('`n_visible` should be an integer')
        if n_visible <= 0:
            raise e.ValueError('`n_visible` should be > 0')

        self._n_visible = n_visible

    @property
    def n_hidden(self):
        """list: List of hidden units.

        """

        return self._n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hidden):
        if not isinstance(n_hidden, list):
            raise e.TypeError('`n_hidden` should be a list')

        self._n_hidden = n_hidden

    @property
    def n_layers(self):
        """int: Number of layers.

        """

        return self._n_layers

    @n_layers.setter
    def n_layers(self, n_layers):
        if not isinstance(n_layers, int):
            raise e.TypeError('`n_layers` should be an integer')
        if n_layers <= 0:
            raise e.ValueError('`n_layers` should be > 0')

        self._n_layers = n_layers

    @property
    def steps(self):
        """int: Number of steps Gibbs' sampling steps.

        """

        return self._steps

    @steps.setter
    def steps(self, steps):
        if not isinstance(steps, int):
            raise e.TypeError('`steps` should be an integer')
        if steps <= 0:
            raise e.ValueError('`steps` should be > 0')

        self._steps = steps

    @property
    def lr(self):
        """float: Learning rate.

        """

        return self._lr

    @lr.setter
    def lr(self, lr):
        if not (isinstance(lr, float) or isinstance(lr, int)):
            raise e.TypeError('`lr` should be a float or integer')
        if lr < 0:
            raise e.ValueError('`lr` should be >= 0')

        self._lr = lr

    @property
    def momentum(self):
        """float: Momentum parameter.

        """

        return self._momentum

    @momentum.setter
    def momentum(self, momentum):
        if not (isinstance(momentum, float) or isinstance(momentum, int)):
            raise e.TypeError('`momentum` should be a float or integer')
        if momentum < 0:
            raise e.ValueError('`momentum` should be >= 0')

        self._momentum = momentum

    @property
    def decay(self):
        """float: Weight decay.

        """

        return self._decay

    @decay.setter
    def decay(self, decay):
        if not (isinstance(decay, float) or isinstance(decay, int)):
            raise e.TypeError('`decay` should be a float or integer')
        if decay < 0:
            raise e.ValueError('`decay` should be >= 0')

        self._decay = decay

    @property
    def T(self):
        """float: Temperature factor.

        """

        return self._T

    @T.setter
    def T(self, T):
        if not (isinstance(T, float) or isinstance(T, int)):
            raise e.TypeError('`T` should be a float or integer')
        if T < 0 or T > 1:
            raise e.ValueError('`T` should be between 0 and 1')

        self._T = T

    @property
    def models(self):
        """list: List of models (RBMs).

        """

        return self._models

    @models.setter
    def models(self, models):
        if not isinstance(models, list):
            raise e.TypeError('`models` should be a list')

        self._models = models

    def fit(self, dataset, batch_size=128, epochs=10):
        """Fits a new DBN model.

        Args:
            dataset (torch.utils.data.Dataset | Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error), log pseudo-likelihood and time from the training step.

        """

        # Initializing MSE and pseudo-likelihood as lists
        mse, pl = [], []

        # Initializing the dataset's variables
        samples, targets, transform = dataset.data.numpy(), dataset.targets.numpy(), dataset.transform

        # For every possible model (RBM)
        for i, model in enumerate(self.models):
            logger.info(f'Fitting layer {i+1}/{self.n_layers} ...')

            # Creating the dataset
            d = Dataset(samples, targets, transform)

            # Fits the RBM
            model_mse, model_pl = model.fit(d, batch_size, epochs)

            # Appending the metrics
            mse.append(model_mse)
            pl.append(model_pl)

            # If the dataset has a transform
            if d.transform:
                # Applies the transform over the samples
                samples = d.transform(d.data)
                # samples = d.transform(d.data)
            
            # If there is no transform
            else:
                # Just gather the samples
                samples = d.data

            # Reshape the samples into an appropriate shape
            samples = samples.view(len(dataset), model.n_visible)

            # Gathers the targets
            targets = d.targets

            # Gathers the transform callable from current dataset
            transform = None

            # Performs a forward pass over the samples
            samples, _ = model.hidden_sampling(d.data)

            # Detaches the variable from the computing graph
            samples = samples.detach()            

        return mse, pl


    def reconstruct(self, dataset, batch_size=128):
        """Reconstruct batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        # For every possible RBM
        for i, rbm in enumerate(self.models):
            logger.info(f'Reconstructing layer {i+1}/{self.n_layers} ...')