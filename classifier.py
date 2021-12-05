#These libraries help to interact with the operating system and the runtime environment respectively
import os
import sys

#Model/Training related libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Dataloader libraries
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score

## Model Architecture definition

class MLP(nn.Module):
    # define model elements
    def __init__(self, size):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(size) - 2):
            layers.append(nn.Linear(size[i], size[i + 1]))
            # layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(size[-2], size[-1]))
        self.model = nn.Sequential(*layers)

        self.embed = nn.Sequential(
            nn.Linear(102400, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x), self.embed(x)
