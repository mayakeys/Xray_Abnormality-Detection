"""
Offline contrast augmentation for cached image tensors.

This script:
- Loads preprocessed image tensors (.pt)
- Applies fixed contrast adjustment(s)
- Saves augmented tensors with descriptive filenames

This is offine augmentation (not applied during training).
"""

import os
import torch
import torchvision.transforms.functional as F

INPUT_DIR = "./input_images/train"
OUTPUT_DIR = "./input_images/train_contrast"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTRAST_FACTORS = [1.5]  # >1 increases contrast

# Helper Functions
def tensor_to_pil(tensor):
    """Convert (C, H, W) tensor to PIL image."""
    return F.to_pil_image(tensor.cpu())

def pil_to_tensor(image):
    """Convert PIL image back to tensor."""
    return F.to_tensor(image)


# Process tensors
for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".pt"):
        continue

    base_name = fname.replace(".pt", "")
    input_path = os.path.join(INPUT_DIR, fname)

    image_tensor = torch.load(input_path)

    # Expect shape (C, H, W)
    if image_tensor.ndim != 3:
        print(f"Skipping {fname}: invalid shape {image_tensor.shape}")
        continue

    image_pil = tensor_to_pil(image_tensor)

    for factor in CONTRAST_FACTORS:
        suffix = f"_contrast_{factor:.1f}".replace(".", "_")
        output_name = f"{base_name}{suffix}.pt"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # Skip if already processed
        if os.path.exists(output_path):
            continue

        contrasted = F.adjust_contrast(image_pil, factor)
        contrast_tensor = pil_to_tensor(contrasted)

        torch.save(contrast_tensor, output_path)
        print(f"Saved contrast tensor: {output_path}")
