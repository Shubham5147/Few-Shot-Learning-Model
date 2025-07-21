"""
data.py
Module for data preparation and loading for few-shot learning.
"""

from typing import List, Tuple
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms




def get_few_shot_dataloader(
    classes: List[int] = [0, 1],
    num_shots: int = 5,
    batch_size: int = 10,
    train: bool = True
) -> Tuple[DataLoader, List[int], transforms.Compose]:
    """
    Prepare a few-shot DataLoader for the specified classes and shots.

    Args:
        classes (List[int]): List of class indices to include.
        num_shots (int): Number of samples per class.
        batch_size (int): Batch size for DataLoader.
        train (bool): Whether to load the training set.

    Returns:
        DataLoader: Few-shot DataLoader.
        List[int]: Classes used.
        transforms.Compose: Transform used.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )
    indices = [
        idx
        for cls in classes
        for idx, label in enumerate(dataset.targets)
        if label == cls
    ][: num_shots * len(classes)]
    few_shot_dataset = Subset(dataset, indices)
    dataloader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=True)
    return dataloader, classes, transform 