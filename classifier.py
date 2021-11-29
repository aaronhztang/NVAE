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
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(102400, 10),
        )
        self.embed = nn.Sequential(
            nn.ReLU(),
            nn.Linear(102400, 2),
        )

    def forward(self, x):
        return self.model(x), self.embed(x)
