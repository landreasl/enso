from ColorMNIST import ColorMNIST
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, dataset
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


mini_batch = 400 #Batchsize from paper
M = 30
learn_rate = 0.005

#TODO implement dataloader class
train_data = enso_loader()
test_data = enso_loader()

train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=True)

data_iter = iter(train_loader)
