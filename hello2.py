# to install pyro, we can use following way in pip (!pip install pyro-ppl)

from typing import Literal, Union
import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, HMC
import plotly.express as px

# Generate some synthetic data
num_data = 5000
x_data = torch.randn((num_data, 1))
w_true = 0.0
y_data = w_true ** 2 * x_data + (torch.randn_like(x_data) * 0.5)  # Some Gaussian noise

# Define the loss as MSE
def loss(w, x, y):
    pred = w ** 2 * x
    return ((pred - y) ** 2).sum()

# Define the model corresponding to the loss
def model(beta, x, y):
    w = pyro.sample("w", dist.Normal(torch.zeros(1), torch.ones(1)*100))  # Prior for w
    pyro.factor("potential", -beta * loss(w, x, y))

# Optimal temperature
beta = 1.0 / np.log(num_data)

# Set up HMC
hmc_kernel = HMC(lambda: model(beta, x_data, y_data), step_size=0.01, num_steps=20)
mcmc = MCMC(hmc_kernel, num_samples=5000, warmup_steps=500)
mcmc.run()

# Extract samples
samples = mcmc.get_samples()["w"]

losses = np.array([loss(w, x_data, y_data) for w in samples])
# Note: if you don't know w_true, you can just find the minimum of loss()
lc = (np.mean(losses) - loss(w_true, x_data, y_data)) / np.log(num_data)

print(lc)