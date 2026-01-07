"""
EfficientNet models for CWT scalogram image classification.

Supports multiple EfficientNet variants (B0, B5, etc.) using pretrained weights
from torchvision. Accepts RGB images of variable sizes thanks to adaptive pooling.
"""

import torch
from torch import nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
)
from typing import Optional, List, Dict, Callable, Tuple

from src.models import BaseModel, register_model


# Mapping of variant names to (model_fn, weights_class) tuples
EFFICIENTNET_VARIANTS: Dict[str, Tuple[Callable, type]] = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights),
    "b5": (efficientnet_b5, EfficientNet_B5_Weights),
}


def _create_efficientnet_model(
    variant: str,
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.5,
    fc_hidden_sizes: Optional[List[int]] = None,
    backbone_freeze: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    """
    Create an EfficientNet model with custom classifier head.
    
    Args:
        variant: EfficientNet variant (e.g., "b0", "b5")
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights
        dropout: Dropout rate for classifier head
        fc_hidden_sizes: List of hidden layer sizes for classifier head
        backbone_freeze: If True, freeze backbone parameters
    
    Returns:
        Tuple of (backbone, classifier) modules
    """
    if variant not in EFFICIENTNET_VARIANTS:
        raise ValueError(
            f"Unknown EfficientNet variant: {variant}. "
            f"Supported variants: {list(EFFICIENTNET_VARIANTS.keys())}"
        )
    
    if fc_hidden_sizes is None:
        fc_hidden_sizes = [128]
    
    model_fn, weights_class = EFFICIENTNET_VARIANTS[variant]
    
    # Load pretrained backbone
    if pretrained:
        weights = weights_class.IMAGENET1K_V1
        backbone = model_fn(weights=weights)
    else:
        backbone = model_fn(weights=None)
    
    # Freeze backbone if requested
    if backbone_freeze:
        for param in backbone.parameters():
            param.requires_grad = False
    
    # Get the number of features from the classifier
    num_features = backbone.classifier[1].in_features
    
    # Remove the original classifier
    backbone.classifier = nn.Identity()
    
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
    
    classifier = nn.Sequential(*fc_layers)
    
    return backbone, classifier


@register_model("efficientnet_b0")
class EfficientNetB0Model(BaseModel):
    """
    EfficientNet-B0 model with pretrained weights for image classification.
    
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
        backbone_freeze: bool = False,
    ):
        """
        Initialize EfficientNet-B0 model.
        
        Args:
            num_classes: Number of output classes (default: 7)
            pretrained: Whether to use pretrained ImageNet weights (default: True)
            dropout: Dropout rate for classifier head (default: 0.5)
            fc_hidden_sizes: List of hidden layer sizes for classifier head.
                           If None, uses [128] as default.
                           Example: [256, 128] for two hidden layers.
            backbone_freeze: If True, freeze backbone parameters (default: False)
        """
        super().__init__()
        self.backbone, self.classifier = _create_efficientnet_model(
            variant="b0",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            fc_hidden_sizes=fc_hidden_sizes,
            backbone_freeze=backbone_freeze,
        )
    
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
        # Adaptive pooling handles any spatial size (224×224, etc.)
        x = self.backbone(x)
        
        # Forward through classifier head
        x = self.classifier(x)
        
        return x


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
        backbone_freeze: bool = False,
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
            backbone_freeze: If True, freeze backbone parameters (default: False)
        """
        super().__init__()
        self.backbone, self.classifier = _create_efficientnet_model(
            variant="b5",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            fc_hidden_sizes=fc_hidden_sizes,
            backbone_freeze=backbone_freeze,
        )
    
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
        x = self.backbone(x)
        
        # Forward through classifier head
        x = self.classifier(x)
        
        return x

