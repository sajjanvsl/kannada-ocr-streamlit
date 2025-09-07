# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 13:54:02 2025

@author: Admin
"""

import torch
import torch.nn as nn

# --- Model Definition ---
class KannadaCNN(nn.Module):
    def __init__(self, num_classes):
        super(KannadaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
