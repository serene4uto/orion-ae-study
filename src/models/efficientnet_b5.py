"""
EfficientNet-B5 model for CWT scalogram image classification.

Uses pretrained weights from torchvision. Accepts RGB images of variable sizes
thanks to adaptive pooling.
"""

import torch
from torch import nn
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from typing import Optional, List

from src.models import BaseModel, register_model


@register_model("efficientnet_b5")
class EfficientNetB5Model(BaseModel):
    """
    EfficientNet-B5 model with pretrained weights for image classification.
    
    Input shape: (batch_size, channels, height, width) - standard PyTorch format
                 Typical: (batch_size, 3, 224, 224) for RGB images
    Output shape: (batch_size, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.5,
        fc_hidden_sizes: Optional[List[int]] = None,
    ):
        """
        Initialize EfficientNet-B5 model.
        
        Args:
            num_classes: Number of output classes (default: 7)
            pretrained: Whether to use pretrained ImageNet weights (default: True)
            dropout: Dropout rate for classifier head (default: 0.5)
            fc_hidden_sizes: List of hidden layer sizes for classifier head.
                           If None, uses [128] as default.
                           Example: [256, 128] for two hidden layers.
        """
        super().__init__()
        
        if fc_hidden_sizes is None:
            fc_hidden_sizes = [128]
        
        # Load pretrained EfficientNet-B5 backbone
        if pretrained:
            weights = EfficientNet_B5_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b5(weights=weights)
        else:
            self.backbone = efficientnet_b5(weights=None)
        
        # Get the number of features from the classifier
        # EfficientNet-B5 classifier has: dropout -> linear(2048, num_classes)
        # The feature extractor outputs 2048 features
        num_features = self.backbone.classifier[1].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Build custom classifier head
        fc_layers = []
        fc_input_size = num_features
        
        for hidden_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(fc_input_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
            fc_input_size = hidden_size
        
        # Final layer to num_classes
        fc_layers.extend([
            nn.Dropout(dropout),
            nn.Linear(fc_input_size, num_classes)
        ])
        
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               Standard PyTorch format: (B, C, H, W)
               Typical: (batch_size, 3, 224, 224) for RGB images
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Ensure input is float and normalized [0, 1] if it's uint8
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Forward through EfficientNet backbone
        # Adaptive pooling handles any spatial size (224×224, 456×456, etc.)
        # (batch_size, channels, height, width) -> (batch_size, 2048)
        x = self.backbone(x)
        
        # Forward through classifier head
        # (batch_size, 2048) -> (batch_size, num_classes)
        x = self.classifier(x)
        
        return x

