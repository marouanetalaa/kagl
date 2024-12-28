# coding: utf-8

# External imports
import torch

from torchtmpl.models.unet import UNet

# Local imports
from . import build_model


def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_cnn():
    cfg = {"class": "VanillaCNN", "num_layers": 4}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")

def test_simple_segmentation_model():
    """
    Test the SimpleSegmentationModel with a dummy input.
    """
    num_classes = 2
    input_size = (1, 256, 256)  # Grayscale image patch
    batch_size = 4

    cfg = {"class": "SimpleSegmentationModel"}

    # Build the model
    model = build_model(cfg,input_size, num_classes)

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, *input_size)  # Shape: (4, 1, 256, 256)

    # Forward pass
    output = model(input_tensor)

    # Print shapes
    print(f"Input tensor shape: {input_tensor.shape}")  # Expected: (4, 1, 256, 256)
    print(f"Output tensor shape: {output.shape}")       # Expected: (4, 2, 256, 256)


def test_unet_model():
    """
    Test the U-Net model with a dummy input.
    """
    input_channels = 1  # Grayscale images
    num_classes = 1     # Single channel output for segmentation
    input_size = (1, 256, 256)  # Grayscale image patch
    batch_size = 4



    # Build the model
    model = UNet(input_channels=input_channels, num_classes=num_classes)

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, *input_size)  # Shape: (4, 1, 256, 256)

    # Forward pass
    output = model(input_tensor)

    # Print shapes
    print(f"Input tensor shape: {input_tensor.shape}")  # Expected: (4, 1, 256, 256)
    print(f"Output tensor shape: {output.shape}")       # Expected: (4, 1, 256, 256)


if __name__ == "__main__":
    # Test the U-Net model
    test_unet_model()


    
