import torch
import torch.optim as optim

# Initialize model, loss, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        features = model(images)
        prototypes = compute_prototypes(features, labels, num_classes=2)
        
        # Compute distances to prototypes
        distances = torch.cdist(features.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        outputs = -distances  # Negative distance for classification
        
        # Loss and optimization
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")