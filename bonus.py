from xRFM.xrfm import xRFM
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import math

np.set_printoptions(threshold=sys.maxsize)

TARGET="subject#"

# part ii) dataset
# df = pd.read_csv('parkinsons.csv', sep=',') 
# X = np.array(df.drop(columns=TARGET).values, np.float32)
# y = np.array(df[TARGET].values, np.float32)

# part iii) dataset
def generate_dataset(n=400):
    np.random.seed(0)

    X = np.random.randn(n, 2)

    # two orthogonal directions
    v1 = np.array([1.0, 0.0])   # x-axis
    v2 = np.array([0.0, 1.0])   # y-axis

    y = np.zeros(n)

    # splitting input space
    mask = X[:, 0] > 0

    # low noise subspace
    y[mask] = X[mask] @ v1 + 0.1 * np.random.randn(mask.sum())

    # high noise subspace
    y[~mask] = 1.5 * X[~mask] @ v2 + 2.5 * np.random.randn((~mask).sum())

    return X.astype(np.float32), y.astype(np.float32)

X, y= generate_dataset()

NOFEATURES = len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# create neural net 
model = nn.Sequential(nn.Linear(X_train.shape[1], NOFEATURES), nn.ReLU(), nn.Linear(NOFEATURES, 1))
# setup loss functions and optimisers
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def eigenvector_angle(v1: torch.Tensor, v2: torch.Tensor):
    # ensure 1D
    v1 = v1.flatten()
    v2 = v2.flatten()

    # get cosine similarity
    cos_sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # derive angle between top eigenvectors
    angle = torch.acos(torch.abs(cos_sim)) * 180.0 / torch.pi

    return angle.item(), cos_sim.item()


# function to calculate and plot the projections and medians of both AGOPs
def calculate_agops(grads: torch.Tensor, residuals: torch.tensor, epoch: int) -> None:
    AGOP = grads.T @ grads

    w = residuals ** 2 # phi(r) = r^2
    w = w / (w.sum() + 1e-8) # normalising all residuals

    # taking square-root of residual means implictly multiplying by residual in outer product
    grads_weighted = grads * torch.sqrt(w).unsqueeze(1)
    AGOP_res = grads_weighted.T @ grads_weighted

    _, eigvecs_std = torch.linalg.eigh(AGOP)
    _, eigvecs_res = torch.linalg.eigh(AGOP_res)
    v_std = eigvecs_std[:, -1]
    v_res = eigvecs_res[:, -1]

    # getting projections of training data onto top eigenvector
    projections_std = X_train @ v_std
    median_std = projections_std.median()
    projections_res = X_train @ v_res
    median_res = projections_res.median()

    angle, cos_sim = eigenvector_angle(v_std, v_res)
    print(angle, cos_sim)

    # plot projections
    # ----------------------------------
    # x-axis is indexed by instances
    indices = np.arange(len(projections_std))

    plt.figure(figsize=(10, 6))

    # scatter plots
    plt.scatter(indices, projections_res.detach().numpy(), label="Residual Projection", alpha=0.4, c="blue")
    plt.scatter(indices, projections_std.detach().numpy(), label="Std Projection", alpha=0.4, c="orange")

    # median lines
    plt.axhline(median_std.detach().numpy(), label="Std Median", c="red")
    plt.axhline(median_res.detach().numpy(), label="Residual Median", c="blue")

    plt.xlabel("Instance Index")
    plt.ylabel("Projection Value (vᵀx)")
    plt.title(f"Projection of Data onto AGOP Eigenvectors (iteration = {epoch})")
    plt.legend()
    plt.grid(True)

    plt.show()

# training loop
# every 50 epochs, we re-test the model projections
for epoch in range(200):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        grads_init = []
        residuals = []

        # calculate gradients for AGOP
        for i in range(len(X_train)):
            # track operations on x_train
            xi = X_train[i].clone().detach().requires_grad_(True)
            # feed values into model
            yi = model(xi)
            # chain rule
            yi.backward() 
            # append gradients and residuals
            grads_init.append(xi.grad.detach())
            residuals.append(y_train[i].squeeze() - yi.squeeze())

        grads_init = torch.stack(grads_init) # reshape
        residuals = torch.stack(residuals)
        calculate_agops(grads_init, residuals, epoch)
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")




