from abc import ABC
import torch
from torch import nn

MODEL_REGISTRY = {}

class BaseModel(nn.Module, ABC):
    """Base class for all models"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError

def register_model(name: str):
    """
    Decorator to automatically register a model class.
    
    Usage:
        @register_model("dummy")
        class DummyModel(BaseModel):
            ...
    """
    def decorator(model_class: BaseModel):

        if name in MODEL_REGISTRY:
            raise ValueError(f"Model name '{name}' is already registered")
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model {model_class.__name__} must inherit from BaseModel")

        MODEL_REGISTRY[name] = model_class
        return model_class # return the model class so it can be used as a decorator
    
    return decorator

def get_model(model_name: str, **params) -> BaseModel:
    """Get a model by name and return an instance of the model"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found")
    return MODEL_REGISTRY[model_name](**params)

def list_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())

@register_model("dummy")
class DummyModel(BaseModel):
    """Dummy model for testing"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


@register_model("simple_cnn")
class SimpleCNN(BaseModel):
    """
    Simple 1D CNN for time series classification.
    
    Input shape: (batch_size, time_steps, channels)
    Output shape: (batch_size, num_classes)
    """
    
    def __init__(
        self,
        input_shape: tuple = None,  # (time_steps, channels)
        num_classes: int = 7,
        num_filters: list = None,  # List of filter sizes for each conv layer
        kernel_sizes: list = None,  # List of kernel sizes for each conv layer
        dropout: float = 0.5,
    ):
        super().__init__()
        
        if input_shape is None:
            raise ValueError("input_shape must be provided")
        
        time_steps, num_channels = input_shape
        
        # Default architecture
        if num_filters is None:
            num_filters = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        
        if len(num_filters) != len(kernel_sizes):
            raise ValueError("num_filters and kernel_sizes must have the same length")
        
        # Build convolutional layers
        conv_layers = []
        in_channels = num_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce time dimension by half
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the output size after convolutions
        # We need to compute the size after all conv and pooling operations
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, time_steps)
            dummy_output = self.conv_layers(dummy_input)
            conv_output_size = dummy_output.numel() // dummy_output.shape[0]  # Flattened size per sample
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, channels)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Transpose from (batch, time, channels) to (batch, channels, time) for Conv1d
        x = x.transpose(1, 2)  # (batch_size, channels, time_steps)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x



