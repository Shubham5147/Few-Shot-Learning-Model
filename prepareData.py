import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Simulate few-shot: select 5 images per class (e.g., classes 0 and 1)
classes = [0, 1]  # e.g., airplane and automobile
num_shots = 5
indices = []
for cls in classes:
    cls_indices = [i for i, label in enumerate(trainset.targets) if label == cls][:num_shots]
    indices.extend(cls_indices)
few_shot_dataset = Subset(trainset, indices)
dataloader = DataLoader(few_shot_dataset, batch_size=10, shuffle=True)