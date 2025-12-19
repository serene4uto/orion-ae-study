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
    Simple 2D CNN for time series classification.
    
    Uses stride=2 in conv layers for wavelet-like downsampling (filter bank + decimation).
    Applies global pooling at the end to collapse spatial dimensions.
    
    Input shape: (batch_size, channels, 1, time_steps) - Conv2d format
    Output shape: (batch_size, num_classes)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 7,
        num_filters: list = None,  # List of filter sizes for each conv layer
        kernel_sizes: list = None,  # List of kernel sizes for each conv layer (will be converted to 2D)
        strides: list = None,  # List of strides for each conv layer (for downsampling)
        dropout: float = 0.5,
        global_pool: str = 'avg',  # 'avg' or 'max' for global pooling
        fc_hidden_sizes: list = None,  # List of hidden layer sizes for FC layers, e.g. [128] or [256, 128]
    ):
        super().__init__()
        
        # Default architecture
        if num_filters is None:
            num_filters = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if strides is None:
            strides = [2, 2, 2]
        if fc_hidden_sizes is None:
            fc_hidden_sizes = [128]  # Default: single hidden layer of size 128
        
        if len(num_filters) != len(kernel_sizes) or len(num_filters) != len(strides):
            raise ValueError("num_filters, kernel_sizes, and strides must have the same length")
        
        # Build convolutional layers with stride-based downsampling (wavelet-like)
        # Using Conv2d: kernel is (height, width) = (1, kernel_size) for time series
        conv_layers = []
        current_channels = in_channels
        
        for i, (out_channels, kernel_size, stride) in enumerate(zip(num_filters, kernel_sizes, strides)):
            # Convert 1D kernel size to 2D: (1, kernel_size) for Conv2d
            kernel_2d = (1, kernel_size)
            stride_2d = (1, stride)
            padding_2d = (0, kernel_size // 2)  # No padding in height, same padding in width
            
            conv_layers.extend([
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_2d,
                    stride=stride_2d,  # Stride-based downsampling (like DWT decimation)
                    padding=padding_2d
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global pooling to collapse spatial dimensions
        if global_pool == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d(1)  # (batch, channels, 1, 1)
        elif global_pool == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d(1)  # (batch, channels, 1, 1)
        else:
            raise ValueError(f"global_pool must be 'avg' or 'max', got {global_pool}")
        
        # After global pooling: (batch, channels, 1, 1) -> flatten to (batch, channels)
        # The number of channels is the last num_filters value
        final_channels = num_filters[-1]
        
        # Build fully connected layers
        fc_layers = []
        fc_input_size = final_channels
        
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
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, 1, time_steps) from dataset
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Input is (batch, channels, 1, time_steps) from dataset - ready for Conv2d
        
        # Apply convolutional layers with stride-based downsampling
        x = self.conv_layers(x)  # (batch_size, final_channels, 1, reduced_time_steps)
        
        # Global pooling to collapse spatial dimensions
        x = self.global_pool(x)  # (batch_size, final_channels, 1, 1)
        
        # Flatten: (batch_size, final_channels, 1, 1) -> (batch_size, final_channels)
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x


# Import models to register them
from src.models import ae_res_inception_net  # noqa: F401
