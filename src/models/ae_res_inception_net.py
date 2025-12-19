import torch
from torch import nn
from src.models import BaseModel, register_model

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if p is None:
        if isinstance(k, int):
            # k is int
            if isinstance(d, int):
                p = d * (k - 1) // 2
            else:
                # d is tuple, but k is int - use d[0] or average?
                p = d[0] * (k - 1) // 2  # or could use both: (d[0] * (k-1)//2, d[1] * (k-1)//2)
        else:
            # k is tuple
            if isinstance(d, int):
                p = tuple(d * (x - 1) // 2 for x in k)
            else:
                p = tuple(d[i] * (k[i] - 1) // 2 for i in range(len(k)))
    return p

class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):

    def __init__(self, 
        in_channels, 
        bottleneck_channels,
        out_channels_per_block,
        mskernel_sizes = [11, 51, 201] 
    ):
        super().__init__()  

        if in_channels > 1:
            self.bottleneck = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=bottleneck_channels, 
                kernel_size=1,
            )
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = in_channels

        # multi scale
        self.convs = nn.ModuleList([
            nn.Conv2d(bottleneck_channels, out_channels_per_block, (k, 1), padding="same")
            for k in mskernel_sizes
        ])

        # 
        self.max_pool = nn.MaxPool2d(kernel_size=(3,1), stride=1, padding=(1,0))
        self.pool_conv = nn.Conv2d(in_channels, out_channels_per_block, 1)

        total_out_channels = out_channels_per_block * (len(mskernel_sizes) + 1)

        self.bn = nn.BatchNorm2d(total_out_channels)
        self.act = nn.ReLU()

    def forward(self, x):

        x_bottle = self.bottleneck(x)
        
        out_list = [conv(x_bottle) for conv in self.convs]

        out_pool = self.pool_conv(self.max_pool(x))

        out_list.append(out_pool)

        x_out = torch.cat(out_list, dim=1) # concatenate outs along channels

        return self.act(self.bn(x_out))

    @property
    def out_channels(self):
        return self.bn.num_features

class ClassifierHead(BaseModel):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 7,
        dropout: float = 0.5,
        fc_hidden_sizes: list = None,
    ):
        super().__init__()
        
        # Global Average Pooling - collapses spatial dimensions (H, W) to 1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Build fully connected layers
        if fc_hidden_sizes is None:
            fc_hidden_sizes = [128]  # Default: single hidden layer
        
        fc_layers = []
        fc_input_size = in_channels
        
        # Add hidden layers
        for hidden_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(fc_input_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
            fc_input_size = hidden_size
        
        # Final classification layer
        fc_layers.extend([
            nn.Dropout(dropout),
            nn.Linear(fc_input_size, num_classes)
        ])
        
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
        
        Returns:
            Output tensor of shape [batch, num_classes]
        """
        # Global Average Pooling: [batch, channels, H, W] -> [batch, channels, 1, 1]
        x = self.gap(x)
        
        # Flatten: [batch, channels, 1, 1] -> [batch, channels]
        x = x.view(x.size(0), -1)
        
        # Fully connected layers: [batch, channels] -> [batch, num_classes]
        x = self.fc(x)
        
        return x

@register_model("ae_res_inception_net")
class AeResInceptionNet(BaseModel):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 7,  # Add this parameter
        stem_kernel = 15,
        stem_downsampling = 5,
        stem_channels = 32,
        inception_block_num: int = 3,
        inception_kernels = [11, 51, 201],
        inception_bottleneck_channels = 32,
        inception_out_channels_per_block = 32,
        dropout: float = 0.5,
        fc_hidden_sizes: list = None,
    ):
        super().__init__()
        self.inception_block_num = inception_block_num

        self.conv_stem = Conv(
            c1=in_channels, 
            c2=stem_channels,
            k=(stem_kernel,1),
            s=stem_downsampling,
        )

        # Build inception blocks with automatic channel progression
        self.inception_blocks = nn.ModuleList()
        current_channels = stem_channels

        # Default values if not provided
        if inception_bottleneck_channels is None:
            inception_bottleneck_channels = stem_channels // 2
        if inception_out_channels_per_block is None:
            inception_out_channels_per_block = stem_channels // 2

        for i in range(inception_block_num):
            block = InceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=inception_bottleneck_channels,
                out_channels_per_block=inception_out_channels_per_block,
                mskernel_sizes=inception_kernels
            )
            self.inception_blocks.append(block)
            
            # Update current_channels for next block
            current_channels = block.out_channels
        
        # Store final output channels
        self.out_channels = current_channels
        
        # Add classifier head
        self.classifier = ClassifierHead(
            in_channels=self.out_channels,
            num_classes=num_classes,
            dropout=dropout,
            fc_hidden_sizes=fc_hidden_sizes,
        )

    def forward(self, x):
        # Input is (batch, channels, 1, time_steps) from dataset - ready for Conv2d
        
        x = self.conv_stem(x)
        
        for block in self.inception_blocks:
            x = block(x)
        
        # Add classifier
        x = self.classifier(x)
        
        return x

