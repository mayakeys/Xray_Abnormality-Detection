#gridsearch.py
"""
Grid search experiment for single-pathology X-ray regression.

Explores:
- hidden layer size
- number of frozen backbone layers
- learning rate

It is used only for hyperparameter selection.
Final training is performed using train.py.
"""

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score

# Hyperparameter grid
hidden_sizes = [256, 512]
frozen_layers = [3, 5]
learning_rates = [5e-4, 1e-4]

# Results dictionary
results = []

# Model definition
class MultiLabelResNet50(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(MultiLabelResNet50, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# Freeze function
def freeze_base_layers(model, until_layer):
    child_counter = 0
    for child in model.base_model.children():
        if child_counter < until_layer:
            for param in child.parameters():
                param.requires_grad = False
        child_counter += 1
    return model

# Grid search loop
for hidden_size, freeze_until, lr in itertools.product(hidden_sizes, frozen_layers, learning_rates):
    print(f"Training model with hidden_size={hidden_size}, freeze_until={freeze_until}, lr={lr}")

    # Initialize model and optimizer
    model = MultiLabelResNet50(num_classes=1, hidden_size=hidden_size).to(device)
    model = freeze_base_layers(model, until_layer=freeze_until)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(3):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = masked_MSE_loss(outputs, labels, class_weights)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = masked_MSE_loss(outputs, labels, class_weights)
            val_loss += loss.item()

            # Custom thresholding
            predicted = torch.where(
                outputs < 0.25, torch.tensor(0.0).to(device),
                torch.where(
                    outputs < 0.75, torch.tensor(0.5).to(device),
                    torch.tensor(1.0).to(device)
                )
            )

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            val_correct += (predicted == labels).sum().item()
            val_total += labels.numel()

    # Compute metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Map 0.0 → 0, 0.5 → 1, 1.0 → 2
    all_preds = (all_preds * 2).astype(int)
    all_labels = (all_labels * 2).astype(int)

    val_accuracy = val_correct / val_total
    val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)
    avg_val_loss = val_loss / len(val_loader)

    # Store results
    results.append({
        'hidden_size': hidden_size,
        'freeze_until': freeze_until,
        'learning_rate': lr,
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1
    })

# Display sorted configs
results.sort(key=lambda x: x['val_f1'], reverse=True)
for r in results:
    print(r)
