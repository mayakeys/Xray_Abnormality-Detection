import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.optim as optim
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model import MultiLabelResNet50 
from dataset import CSVDataset

# --------------------------------------------
# Parameters
# --------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_root = '/central/groups/CS156b/2025/CodeMonkeys/input_images'

target_col = 'Lung Opacity' if os.getenv('TARGET_COL')   is None else os.getenv('TARGET_COL')
image_root_dir = "input_images/train" if os.getenv('IMAGE_ROOT_DIR')   is None else os.getenv('IMAGE_ROOT_DIR')
model_save_dir = 'amb_models/lo_full' if os.getenv('SAVE_DIR')   is None else os.getenv('SAVE_DIR')

frontal_status = True if os.getenv('FRONTAL_STATUS')   is None else bool(os.getenv('FRONTAL_STATUS'))
end_df = None if os.getenv('END_DF')   is None else int(os.getenv('END_DF'))

uncertain_weight_factor = 0.25 if os.getenv('UNCERTAIN_WF')   is None else float(os.getenv('UNCERTAIN_WF'))
neg_cutoff = 0.25 if os.getenv('NEG_CUTOFF')   is None else float(os.getenv('NEG_CUTOFF'))
pos_cutoff = 0.75 if os.getenv('POS_CUTOFF')   is None else float(os.getenv('POS_CUTOFF'))

num_epochs = 30 if os.getenv('NUM_EPOCHS')   is None else int(os.getenv('NUM_EPOCHS'))
hidden_size = 512 if os.getenv('HIDDEN_SIZE')   is None else int(os.getenv('HIDDEN_SIZE'))
lr = 1e-4 if os.getenv('LR')   is None else float(os.getenv('LR'))
weight_decay = 1e-4 if os.getenv('WEIGHT_DECAY')   is None else float(os.getenv('WEIGHT_DECAY'))
freeze_until = 4 if os.getenv('FREEZE_UNTIL')   is None else int(os.getenv('FREEZE_UNTIL'))

train_save_dir = os.path.join(image_root, 'train_lateral') if not frontal_status else os.path.join(image_root, 'train_frontal')

# --------------------------------------------
# Helper Functions
# --------------------------------------------
def get_filtered_df(col, num=None, frontal_status=True):
    full_train_df = pd.read_csv('train2023.csv')

    if num is not None:
        full_train_df = full_train_df.iloc[:num]

    # Filter for Frontal if specified
    if frontal_status:
        full_train_df = full_train_df[full_train_df['Frontal/Lateral'] == 'Frontal']
    else:
        full_train_df = full_train_df[full_train_df['Frontal/Lateral'] == 'Lateral']

    # Drop rows with missing label in the given column
    filtered_train_df = full_train_df.dropna(subset=[col]).copy()

    # Normalize label values from {-1, 0, 1} to {0, 0.5, 1}
    filtered_train_df[col] = (filtered_train_df[col] + 1) / 2

    return filtered_train_df

def freeze_base_layers(model, until_layer):
    """
    Freeze layers of ResNet-50 up to a certain stage (e.g., until_layer=6 means keep layers 0-5 frozen).
    """
    for i, child in enumerate(model.model.children()):
        if i < until_layer:
            for p in child.parameters():
                p.requires_grad = False
    return model

criterion = nn.MSELoss(reduction="none")

def masked_MSE_loss(output, target):
    mask = ~torch.isnan(target)
    loss = criterion(output, target)
    # Return mean loss for valid entries
    return (loss * mask).sum() / mask.sum()

# --------------------------------------------
# Data Preparation and Train/Val Split
# --------------------------------------------
filtered_train_df = get_filtered_df(target_col, num=end_df, frontal_status=frontal_status)
label_counts = filtered_train_df[target_col].value_counts()

target_columns = [target_col]
train_df, val_df = train_test_split(filtered_train_df, test_size=0.15, random_state=42)

# Create training dataset
train_dataset = CSVDataset(
    dataframe=train_df, 
    image_root_dir=image_root, 
    target_columns=target_columns, 
    save_dir=train_save_dir, 
    use_saved_images=True
)

# Create validation dataset
val_dataset = CSVDataset(
    dataframe=val_df, 
    image_root_dir=image_root, 
    target_columns=target_columns, 
    save_dir=train_save_dir, 
    use_saved_images=True
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Handle Class Imbalances with Sample Weights

# Map continuous labels to discrete bins for sampling
# 0.0 → negative, 0.5 → uncertain, 1.0 → positive
lo_labels = train_df[target_col].values
label_map = {0.0: 0, 0.5: 1, 1.0: 2}
mapped_labels = np.array([label_map[float(lbl)] for lbl in lo_labels])

# Inverse-frequency weights per class
class_counts = np.bincount(mapped_labels)
weights = 1. / (class_counts + 1e-6)

# Assign each sample a weight based on its label
sample_weights = torch.tensor(weights[mapped_labels], dtype=torch.float)

# Weighted sampler to oversample rare classes
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)

# --------------------------------------------
# Training
# --------------------------------------------

# Prepare directory and log file
os.makedirs(model_save_dir, exist_ok=True)
log_file_path = os.path.join(model_save_dir, "training_log.txt")
conf_path = os.path.join(model_save_dir, "conf_matrices/")
os.makedirs(conf_path, exist_ok=True)

with open(log_file_path, 'w') as f:
    f.write(f"Filtered train DataFrame length: {len(filtered_train_df)}\n")

# Hyperparameters and model setup
model = MultiLabelResNet50(hidden_size=hidden_size).to(device)
model = freeze_base_layers(model, until_layer=freeze_until)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Early stopping parameters
early_stopping_patience = 3
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Compute masked regression loss (sampling handles imbalance)
        loss = masked_MSE_loss(outputs, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        predicted_class = torch.where(
            outputs < neg_cutoff, torch.tensor(0.0).to(device),
            torch.where(
                outputs < pos_cutoff, torch.tensor(0.5).to(device),
                torch.tensor(1.0).to(device)
            )
        )

        correct += (predicted_class == labels).sum().item()
        total += labels.numel()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    with open(log_file_path, "a") as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Validation phase
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
            loss = masked_MSE_loss(outputs, labels)
            val_loss += loss.item()

            predicted_class = torch.where(
                outputs < neg_cutoff, torch.tensor(0.0).to(device),
                torch.where(
                    outputs < pos_cutoff, torch.tensor(0.5).to(device),
                    torch.tensor(1.0).to(device)
                )
            )

            all_preds.append(predicted_class.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            val_correct += (predicted_class == labels).sum().item()
            val_total += labels.numel()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    with open(log_file_path, "a") as f:
        f.write(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")

    # Convert float predictions to int for confusion matrix
    float_to_int = {0.0: 0, 0.5: 1, 1.0: 2}
    all_preds_np = np.array([float_to_int[val] for val in np.concatenate(all_preds).flatten()])
    all_labels_np = np.array([float_to_int[val] for val in np.concatenate(all_labels).flatten()])

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0.0 (Neg)", "0.5 (Unc)", "1.0 (Pos)"],
                yticklabels=["0.0 (Neg)", "0.5 (Unc)", "1.0 (Pos)"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Validation Confusion Matrix (Epoch {epoch+1})")

    cm_path = os.path.join(conf_path, f"epoch_{epoch+1}.png")
    plt.savefig(cm_path)
    plt.close()

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        with open(log_file_path, "a") as f:
            f.write("⛔ Early stopping triggered.\n")
        break

    torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch_{epoch+1}.pth"))