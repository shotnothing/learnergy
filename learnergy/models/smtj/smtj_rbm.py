"""SMTJ Bernoulli-Bernoulli Restricted Boltzmann Machine.

Attributes:
    logger (TYPE): Learnergy's logging utility.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm

from learnergy.utils import logging

from learnergy.models.bernoulli.rbm import RBM

logger = logging.get_logger(__name__)


class SMTJRBM(RBM):
    """This class is strongly based upon Learnergy's bernouli.RBM class but
    modified to incorporate sigma ratio, shift and slope.
    
    Attributes:
        a (Tensor): Trainable bias for the visible layer
        b (Tensor): Trainable bias for the hidden layer
        lr (Float): Learning rate
        n_hidden (Integer): Number of hidden nodes
        n_visible (Integer): Number of visible nodes
        optimizer (Optimizer): Optimizer used in learning
        sigma_initial_shift (Float): SMTJ property that gives each node a slightly different sigmoid activation function shift
        sigma_initial_slope (Float): SMTJ property that gives each node a slightly different sigmoid activation function slope
        sigma_ratio (Float): SMTJ property that affects the sampling
        steps (Integer): The K in CD-K
        W (Tensor): Trainable weights in the RBM
    """

    def __init__(
        self,
        n_visible: Optional[int] = 128,
        n_hidden: Optional[int] = 128,
        steps: Optional[int] = 1,
        learning_rate: Optional[float] = 0.1,
        momentum: Optional[float] = 0.0,
        decay: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
        use_gpu: Optional[bool] = False,
        sigma_ratio: Optional[float] = 0.5,
        sigma_initial_shift: Optional[float] = 0.3,
        sigma_initial_slope: Optional[float] = 0.3
    ) -> None:

        logger.info("Overriding class: Model -> RBM.")

        super(SMTJRBM, self).__init__(use_gpu=use_gpu)

        # Amount of visible units
        self.n_visible = n_visible

        # Amount of hidden units
        self.n_hidden = n_hidden

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter -> not used in our case, leave at 0.0
        self.momentum = momentum

        # Weight decay -> not used in our case, leave at 0.0
        self.decay = decay

        # Temperature factor -> not used in our case, leave at 1.0
        self.T = temperature

        # Weights matrix
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)

        # Visible units bias
        self.a = nn.Parameter(torch.zeros(n_visible))

        # Hidden units bias
        self.b = nn.Parameter(torch.zeros(n_hidden))

        self.sigma_ratio = sigma_ratio
        self.sigma_initial_shift = sigma_initial_shift
        self.sigma_initial_slope = sigma_initial_slope

        # Creating the optimizer object
        self.optimizer = opt.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay
        )

        logger.info("Class overrided.")
        logger.debug(
            "Size: (%d, %d) | Learning: CD-%d | "
            "Hyperparameters: lr = %s, momentum = %s, decay = %s, T = %s.",
            self.n_visible,
            self.n_hidden,
            self.steps,
            self.lr,
            self.momentum,
            self.decay,
            self.T,
        )

    def setup_slope_shift(self):
        """Initialize the random slope and shift.
        """
        self.v_slope = nn.Parameter(
            torch.abs(torch.randn(self.n_visible) * self.sigma_initial_slope + 1),
            requires_grad=False)

        self.v_shift = nn.Parameter(
            torch.randn(self.n_visible) * self.sigma_initial_shift,
            requires_grad=False)

        self.h_slope = nn.Parameter(
            torch.abs(torch.randn(self.n_hidden) * self.sigma_initial_slope + 1),
            requires_grad=False)

        self.h_shift = nn.Parameter(
            torch.randn(self.n_hidden) * self.sigma_initial_shift,
            requires_grad=False)

    def sample_from_p(self, p: torch.Tensor) -> torch.Tensor:
        """Sample the data with bernouli distrabution, but affected by sigma ratio

        Args:
            p (Tensor): Input data to be sampled

        Returns:
            Tensor: Bernouli sampled data, affected by sigma ratio
        """
        p = torch.randn(p.shape) * self.sigma_ratio * (0.5 - torch.abs(p - 0.5)) + p
        return F.relu(torch.sign(p - torch.autograd .Variable(torch.rand(p.size()))))

    def pre_activation(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> torch.Tensor:
        """Performs the pre-activation over hidden neurons, i.e., Wx' + b.

        Args:
            v (Tensor): A tensor incoming from the visible layer.
            scale (Optional[Boolean], optional): A boolean to decide whether temperature
            should be used or not.

        No Longer Returned:
            (Tensor): An input for any type of activation function.

        """

        # Calculating neurons' activations
        activations = F.linear(
            v,
            (self.W * self.h_slope).t(),
            self.b * self.h_slope + self.h_shift
        )

        return activations

    def hidden_sampling(
        self, v: torch.Tensor, scale: Optional[bool] = False
    ) -> torch.Tensor:
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (Tensor): A tensor incoming from the visible layer.
            scale (Optional[Boolean], optional): A boolean to decide whether temperature should be used or not.

        No Longer Returned:
            (Tensor): The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(
            v,
            (self.W * self.h_slope).t(),
            self.b * self.h_slope + self.h_shift
        )

        probs = torch.sigmoid(activations)

        # Sampling current states
        states = self.sample_from_p(probs)

        return probs, states

    def visible_sampling(
        self, h: torch.Tensor, scale: Optional[bool] = False
    ) -> torch.Tensor:
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (Tensor): A tensor incoming from the hidden layer.
            scale (Optional[Boolean], optional): A boolean to decide whether temperature should be used or not.

        No Longer Returned:
            (Tensor): The probabilities and states of the visible layer sampling.

        """
        # Calculating neurons' activations
        activations = F.linear(
            h, 
            (self.W.t() * self.v_slope).t(), 
            self.a * self.v_slope + self.v_shift
        )

        probs = torch.sigmoid(activations)

        # Sampling current states
        states = self.sample_from_p(probs)

        return probs, states

    def gibbs_sampling(
        self, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the whole Gibbs sampling procedure.

        Args:
            v (Tensor): A tensor incoming from the visible layer.

        No Longer Returned:
            (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): The probabilities and states of the hidden layer sampling (positive),
                the probabilities and states of the hidden layer sampling (negative)
                and the states of the visible layer sampling (negative).

        """

        # Calculating positive phase hidden probabilities and states
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v)

        # Initially defining the negative phase
        neg_hidden_states = pos_hidden_states

        # Performing the Contrastive Divergence
        for _ in range(self.steps):
            # Calculating visible probabilities and states
            _, visible_states = self.visible_sampling(
                neg_hidden_states, True
            )

            # Calculating hidden probabilities and states
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, True
            )

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_states,
        )

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculates and frees the system's energy.
        
        Args:
            samples (torch.Tensor): Samples to be energy-freed.
        
        Returns:
            torch.Tensor: The system's energy based on input samples.
        
        """

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=1)

        # Calculate the visible term
        v = torch.mv(samples, self.a)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
    ) -> Tuple[float, float]:
        """Fits a new RBM model.
        
        Args:
            dataset (Dataset): A Dataset object containing the training data.
            batch_size (Optional[Integer], optional): Amount of samples per batch.
            epochs (Optional[Integer], optional): Number of training epochs.
        
        No Longer Returned:
            (Tuple[Float, Float]): MSE (mean squared error) and log pseudo-likelihood from the training step.
        
        """

        # Transforming the dataset into training batches
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        self.setup_slope_shift()

        # For every epoch
        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse, pl = 0, 0

            # For every batch
            for samples, _ in tqdm(batches):
                # Flattening the samples' batch
                samples = samples.reshape(len(samples), self.n_visible)

                # Checking whether GPU is avaliable and if it should be used
                if self.device == "cuda":
                    # Applies the GPU usage to the data
                    samples = samples.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Calculates the loss for further gradients' computation
                cost = torch.mean(self.energy(samples)) - torch.mean(
                    self.energy(visible_states)
                )

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating current's batch MSE
                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size
                ).detach()

                # Calculating the current's batch logarithm pseudo-likelihood
                batch_pl = self.pseudo_likelihood(samples).detach()

                # Summing up to epochs' MSE and pseudo-likelihood
                mse += batch_mse
                pl += batch_pl

            # Normalizing the MSE and pseudo-likelihood with the number of batches
            mse /= len(batches)
            pl /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), pl=pl.item(), time=end - start)

            logger.info("MSE: %f | log-PL: %f", mse, pl)

        return mse, pl

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor]:
        """Reconstructs batches of new samples.
        
        Args:
            dataset (Dataset): A Dataset object containing the testing data.
        
        No Longer Returned:
            (Tuple[Float, Tensor]): Reconstruction error and visible probabilities, i.e., P(v|h).
        
        """

        logger.info("Reconstructing new samples ...")

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # For every batch
        for samples, _ in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(len(samples), self.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == "cuda":
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            _, pos_hidden_states = self.hidden_sampling(samples)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(pos_hidden_states)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size
            )

            # Summing up the reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        logger.info("MSE: %f", mse)

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.
        
        Args:
            x (Tensor): An input tensor for computing the forward pass.
        
        Returns:
            Tensor: A tensor containing the RBM's outputs.
        
        """

        # Calculates the outputs of the model
        x, _ = self.hidden_sampling(x)

        return x
