#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:39:25 2021

@author: arevell
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import torch
import time


x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()


x.T


#defining helper functions
def plot_make(r = 1, c = 1, size_length = None, size_height = None, dpi = 300, sharex = False, sharey = False , squeeze = True):
    """
    helper function to make subplots in python quickly
    """
    if size_length == None:
       size_length = 4* c
    if size_height == None:
        size_height = 4 * r
    fig, axes = plt.subplots(r, c, figsize=(size_length, size_height), dpi=dpi, sharex =sharex, sharey = sharey, squeeze = squeeze)
    return fig, axes



#%%
# -*- coding: utf-8 -*-
import numpy as np

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)


# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

t0 = time.time()
learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
t1 = time.time()
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')


y1 = np.array([a + b * x + c * x ** 2 + d * x ** 3 for x in x])

fig, axes = plot_make()
sns.scatterplot(x = x, y= y, ax = axes, linewidth=0, s = 0.1)
sns.scatterplot(x = x, y= y1, ax = axes, linewidth=0, s = 0.1)



#%%

dtype = torch.float
device = torch.device("cuda:0")
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6

for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

fig, axes = plot_make()
sns.scatterplot(x = x, y= y, ax = axes, linewidth=0, s = 0.1)
















