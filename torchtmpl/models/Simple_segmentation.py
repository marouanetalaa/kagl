import torch
import torch.nn as nn

def conv_block(cin, cout, kernel_size=3, padding=1):
    return [
        nn.Conv2d(cin, cout, kernel_size=kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(cout),
    ]

def SimpleSegmentationModel(cfg, input_size, num_classes):
    """
    Now we expect input_size = (C, H, W).
    Extract channels from input_size[0].
    """
    input_channels = input_size[0]

    layers = []
    layers.extend(conv_block(input_channels, 16))  # Block 1
    layers.extend(conv_block(16, 32))              # Block 2
    layers.extend(conv_block(32, 64))              # Block 3
    layers.append(nn.Conv2d(64, 1, kernel_size=1))  # Final 1x1 Conv Layer

    return nn.Sequential(*layers)
