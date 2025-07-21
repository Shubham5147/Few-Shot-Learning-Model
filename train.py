"""
train.py
Script to train the SimpleCNN model using prototypical networks for few-shot learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from prototypical_network import SimpleCNN, compute_prototypes
from data import get_few_shot_dataloader

def main(args):
    # Data
    dataloader, _, _ = get_few_shot_dataloader(classes=args.classes, num_shots=args.num_shots, batch_size=args.batch_size, train=True)

    # Model, loss, optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            prototypes = compute_prototypes(features, labels, num_classes=len(args.classes))
            distances = torch.cdist(features.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
            outputs = -distances  # Negative distance for classification
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleCNN with prototypical networks.")
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1], help='Class indices to use')
    parser.add_argument('--num_shots', type=int, default=5, help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    main(args) 