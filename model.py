#model.py
# --------------------------------------------------
# Model architectures used for X-ray abnormality regression.
# These classes define ONLY the network structure.
# Training and inference logic live elsewhere.
# --------------------------------------------------

import torch.nn as nn
from torchvision import models


class MultiLabelResNet50(nn.Module):
    """
    Standard ResNet-50 regressor with a lightweight MLP head.

    Used for most pathologies. Outputs a single scalar in [0, 1]
    (sigmoid activation), later rescaled at inference time.
    """
    def __init__(self, hidden_size: int = 512, dropout: float = 0.5):
        super().__init__()

        # Pretrained ImageNet backbone
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # Replace the classification head with a regression head
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class MultiLabelResNet50Reg(nn.Module):
    """
    More strongly regularized ResNet-50 variant.

    Uses a smaller hidden layer and higher dropout.
    Employed for pathologies (e.g., Fracture) that benefited
    from additional regularization during training.
    """
    def __init__(self):
        super().__init__()

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        in_feats = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
