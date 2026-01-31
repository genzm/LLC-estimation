# to install pyro, we can use following way in pip (!pip install pyro-ppl)

from typing import Literal, Union
import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, HMC
import plotly.express as px

# Define the loss function
def K(w):
    return w[0] ** 4

# Define the model corresponding to the loss
def model(beta):
    w = pyro.sample("w", dist.Normal(torch.zeros(1), torch.ones(1)*10))  # Prior for 2D w
    pyro.factor("K", -beta * K(w))

# Adjustable inverse temperature
beta = 0.1  # Feel free to change this value

# Set up HMC
hmc_kernel = HMC(lambda: model(beta), step_size=0.0001, num_steps=20)
mcmc = MCMC(hmc_kernel, num_samples=5000, warmup_steps=500)
mcmc.run()

# Extract samples
samples = mcmc.get_samples()["w"]

# Make a histogram
# px.histogram(samples.numpy())

losses = np.array([K(w) for w in samples])
lc = beta * np.mean(losses)

print(lc)