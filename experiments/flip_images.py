"""
Offline horizontal flip augmentation for cached image tensors.

This script:
- Loads preprocessed image tensors (.pt) from a directory
- Applies a deterministic horizontal flip
- Saves flipped tensors to a separate directory

This is offline augmentation (not applied during training).
"""

import os
import torch
import torchvision.transforms as transforms

INPUT_DIR = "./input_images/train"
OUTPUT_DIR = "./input_images/train_flipped"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Flip transform
flip_transform = transforms.RandomHorizontalFlip(p=1.0)

# Process tensors
for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".pt"):
        continue

    input_path = os.path.join(INPUT_DIR, fname)
    output_path = os.path.join(OUTPUT_DIR, fname)

    # Skip if already processed
    if os.path.exists(output_path):
        continue

    image_tensor = torch.load(input_path)

    # Expect shape (C, H, W)
    if image_tensor.ndim != 3:
        print(f"Skipping {fname}: invalid shape {image_tensor.shape}")
        continue

    flipped_tensor = flip_transform(image_tensor)
    torch.save(flipped_tensor, output_path)

    print(f"Saved flipped tensor: {output_path}")
