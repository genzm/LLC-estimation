# to install pyro, we can use following way in pip (!pip install pyro-ppl)

from typing import Literal, Union
import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, HMC
import plotly.express as px


class SGLD(torch.optim.Optimizer):
    r"""
    Implements Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    This optimizer blends Stochastic Gradient Descent (SGD) with Langevin Dynamics,
    introducing Gaussian noise to the gradient updates. It can also include an
    elasticity term that , acting like
    a special form of weight decay.

    It follows Lau et al.'s (2023) implementation, which is a modification of
    Welling and Teh (2011) that omits the learning rate schedule and introduces
    an elasticity term that pulls the weights towards their initial values.

    The equation for the update is as follows:

    $$
    \begin{gathered}
    \Delta w_t=\frac{\epsilon}{2}\left(\frac{\beta n}{m} \sum_{i=1}^m \nabla \log p\left(y_{l_i} \mid x_{l_i}, w_t\right)+\gamma\left(w^_0-w_t\right) - \lambda w_t\right) \\
    +N(0, \epsilon\sigma^2)
    \end{gathered}
    $$

    where $w_t$ is the weight at time $t$, $\epsilon$ is the learning rate,
    $(\beta n)$ is the inverse temperature (we're in the tempered Bayes paradigm),
    $n$ is the number of training samples, $m$ is the batch size, $\gamma$ is
    the elasticity strength, $\lambda$ is the weight decay strength, $n$ is the
    number of samples, and $\sigma$ is the noise term.

    :param params: Iterable of parameters to optimize or dicts defining parameter groups
    :param lr: Learning rate (required)
    :param noise_level: Amount of Gaussian noise introduced into gradient updates (default: 1). This is multiplied by the learning rate.
    :param weight_decay: L2 regularization term, applied as weight decay (default: 0)
    :param elasticity: Strength of the force pulling weights back to their initial values (default: 0)
    :param temperature: Temperature. (default: 1)
    :param num_samples: Number of samples to average over (default: 1)

    Example:
        >>> optimizer = SGLD(model.parameters(), lr=0.1, temperature=torch.log(n)/n)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        - The `elasticity` term is unique to this implementation and serves to guide the
        weights towards their original values. This is useful for estimating quantities over the local
        posterior.
    """

    def __init__(self, params, lr=1e-3, noise_level=1., elasticity=0., temperature: Union[Literal['adaptive'], float]=1., bounding_box_size=None, num_samples=1, batch_size=None):
        defaults = dict(lr=lr, noise_level=noise_level, elasticity=elasticity, temperature=temperature, bounding_box_size=None, num_samples=num_samples, batch_size=batch_size)
        super(SGLD, self).__init__(params, defaults)

        # Save the initial parameters if the elasticity term is set
        for group in self.param_groups:
            if group['elasticity'] != 0:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['initial_param'] = torch.clone(p.data).detach()
            if group['temperature'] == "adaptive":  # TODO: Better name
                group['temperature'] = np.log(group["num_samples"])

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    if p.grad is None:
                        continue

                    dw = p.grad * (group["num_samples"] / group["batch_size"]) / group['temperature']

                    if group['elasticity'] != 0:
                        initial_param = self.state[p]['initial_param']
                        dw.add_((p - initial_param), alpha=group['elasticity'])

                    p.add_(dw, alpha = -0.5 * group['lr'])

                    # Add Gaussian noise
                    noise = torch.normal(mean=0., std=group['noise_level'], size=dw.size(), device=dw.device)
                    p.add_(noise, alpha=group['lr'] ** 0.5)

                    # Clamp to bounding box size
                    if group['bounding_box_size'] is not None:
                        torch.clamp_(p, min=-group['bounding_box_size'], max=group['bounding_box_size'])


# Generate some synthetic data
num_data = 1000
batch_size = 100
num_steps = 5000
x_data = torch.randn((num_data, 1))
w_true = 0.0
y_data = w_true ** 2 * x_data + (torch.randn_like(x_data) * 0.5)  # Some Gaussian noise

# Define the loss as MSE
def loss_fn(w, x, y):
    pred = w ** 2 * x
    return ((pred - y) ** 2).sum()

# Optimal temperature
beta = 1.0 / np.log(num_data)

w_sample = torch.tensor(w_true, requires_grad=True)
optimizer = SGLD([w_sample], lr=0.001, num_samples=num_data, batch_size=batch_size)

samples = []

# Run SGLD
for step in range(num_steps):
    # Store the runs for later plotting
    samples.append(w_sample.detach().cpu().numpy().copy())

    # Sampling batches with replacement for ease of implmentation, but for
    # memory efficiency with large models you probably will be forced to sample
    # without replacement
    batch_indices = torch.randperm(num_data)[:batch_size]
    x_batch = x_data[batch_indices]
    y_batch = y_data[batch_indices]

    optimizer.zero_grad()

    loss = beta * loss_fn(w_sample, x_batch, y_batch)
    loss.backward()

    optimizer.step()

samples = np.array(samples)

# Create a histogram
# px.histogram(samples)


losses = np.array([loss_fn(w, x_data, y_data) for w in samples])
# Note: if you don't know w_true, you can just find the minimum of loss()
lc = (np.mean(losses) - loss_fn(w_true, x_data, y_data)) / np.log(num_data)

print(lc)