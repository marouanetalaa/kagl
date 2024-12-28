# coding: utf-8

# External imports
import torch

# Local imports
from .base_models import *
from .cnn_models import *
from .Simple_segmentation import *
from .unet import *

def build_model(cfg, input_size, num_classes):
    if cfg['class'] == "UNet":
        input_channels = input_size[0]  # Extract input channels from input_size
        return UNet(input_channels=input_channels, num_classes=num_classes)
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")
