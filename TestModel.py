# Test on new images (simulated here with another subset)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_indices = [i for i, label in enumerate(testset.targets) if label in classes][:10]
test_dataset = Subset(testset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        features = model(images)
        prototypes = compute_prototypes(features, labels, num_classes=2)
        distances = torch.cdist(features.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        predictions = distances.argmin(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")