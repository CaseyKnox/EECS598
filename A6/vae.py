from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def hello_vae():
    print("Hello from vae.py!")


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.hidden_dim = None # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        self.hidden_dim = 128
        H_d = self.hidden_dim
        Z = self.latent_size
        H,W = 28,28
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, H_d),
            nn.ReLU(),
            nn.Linear(H_d, H_d),
            nn.ReLU(),
            nn.Linear(H_d, H_d),
        )
        self.mu_layer = nn.Linear(H_d, Z)
        self.logvar_layer = nn.Linear(H_d, Z)
        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        self.decoder = nn.Sequential(
            nn.Linear(Z, H_d),
            nn.ReLU(),
            nn.Linear(H_d, H_d),
            nn.ReLU(),
            nn.Linear(H_d, H_d),
            nn.ReLU(),
            nn.Linear(H_d, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(1, (1,H,W))
        )
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################


    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Encoder
        enc = self.encoder.forward(x)           # (N, H_d)
        mu = self.mu_layer.forward(enc)         # (N, Z)
        logvar = self.logvar_layer.forward(enc) # (N, Z)

        # Latent vector Z
        z = reparametrize(mu, logvar)           # (N, Z)

        # Decoder
        x_hat = self.decoder.forward(z)         # (N, 1, H, W)
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C
        self.hidden_dim = None # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Define a FC encoder as described in the notebook that transforms the image--after  #
        # flattening and now adding our one-hot class vector (N, H*W + C)--into a hidden_dimension #               #
        # (N, H_d) feature space, and a final two layers that project that feature space           #
        # to posterior mu and posterior log-variance estimates of the latent space (N, Z)          #
        ############################################################################################
        M = self.input_size + self.num_classes
        self.hidden_dim = 128
        H_d = self.hidden_dim
        Z = self.latent_size
        C = self.num_classes
        self.encoder = nn.Sequential(
            nn.Linear(M, H_d),
            nn.ReLU(),
            nn.Linear(H_d, H_d),
            nn.ReLU(),
            nn.Linear(H_d, H_d),
        )
        self.mu_layer = nn.Linear(H_d, Z)
        self.logvar_layer = nn.Linear(H_d, Z)

        ############################################################################################
        # TODO: Define a fully-connected decoder as described in the notebook that transforms the  #
        # latent space (N, Z + C) to the estimated images of shape (N, 1, H, W).                   #
        ############################################################################################
        H,W = 28,28
        self.decoder = nn.Sequential(
            nn.Linear(H_d + C, H_d),
            nn.ReLU(),
            nn.Linear(H_d, H_d),
            nn.ReLU(),
            nn.Linear(H_d, H_d),
            nn.ReLU(),
            nn.Linear(H_d, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(1, (1,H,W))
        )
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the concatenation of input batch and one hot vectors through the encoder model  #
        # to get posterior mu and logvariance                                                      #
        # (2) Reparametrize to compute the latent vector z                                         #
        # (3) Pass concatenation of z and one hot vectors through the decoder to resconstruct x    #
        ############################################################################################

        # Prepare input concatenated x + c
        x_flat = x.flatten(start_dim=1) # (N, H*W)
        xc     = torch.cat([x_flat, c], dim=1) # (N, H*W + C)
        # print("x_flat", x_flat.shape)
        # print("c", c.shape)
        # print("xc", xc.shape)

        # Forward Pass
        z = self.encoder.forward(xc)          # (N, Z) latent space of size Z
        mu = self.mu_layer.forward(z)         # (N, Z)
        logvar = self.logvar_layer.forward(z) # (N, Z)

        # Decoder
        zc = torch.cat([z, c], dim=1)   # (N, Z + C)
        # print("zc", zc.shape)
        x_hat = self.decoder.forward(zc) # (N, 1, H, W)

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar



def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    eps = torch.randn_like(mu)
    sigma = torch.exp(0.5 * logvar) # std
    z = sigma * eps + mu
    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    loss = None
    ################################################################################################
    # TODO: Compute negative variational lowerbound loss as described in the notebook              #
    ################################################################################################
    # Reconstruction: sum over pixels per sample, then mean over the batch
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]    # (1,)

    # KL: sum over latent dims per sample, then mean over batch
    kl = -0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar)).sum(dim=1).mean()    # (N,Z)

    loss = recon + kl
    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss

