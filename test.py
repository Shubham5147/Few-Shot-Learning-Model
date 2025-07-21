"""
test.py
Script to test the SimpleCNN model using prototypical networks for few-shot learning.
"""

import torch
import torch.nn as nn
import argparse
from prototypical_network import SimpleCNN, compute_prototypes
from data import get_few_shot_dataloader

def main(args):
    # Data
    test_loader, _, _ = get_few_shot_dataloader(classes=args.classes, num_shots=args.num_shots, batch_size=args.batch_size, train=False)

    # Model (should load trained weights in a real scenario)
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            prototypes = compute_prototypes(features, labels, num_classes=len(args.classes))
            distances = torch.cdist(features.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
            predictions = distances.argmin(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SimpleCNN with prototypical networks.")
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1], help='Class indices to use')
    parser.add_argument('--num_shots', type=int, default=5, help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    args = parser.parse_args()
    main(args) 