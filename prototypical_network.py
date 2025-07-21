"""
prototypical_network.py
Module containing the SimpleCNN model and prototypical network logic for few-shot learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for feature extraction.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 64)  # Feature dimension

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def compute_prototypes(features, labels, num_classes):
    """
    Compute class prototypes as the mean feature vector for each class.
    Args:
        features (Tensor): Feature vectors from the model.
        labels (Tensor): Corresponding class labels.
        num_classes (int): Number of classes.
    Returns:
        Tensor: Prototypes for each class.
    """
    prototypes = torch.zeros(num_classes, features.size(1)).to(features.device)
    for cls in range(num_classes):
        mask = (labels == cls)
        prototypes[cls] = features[mask].mean(dim=0)
    return prototypes 