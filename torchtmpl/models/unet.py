import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, base_features=64):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(input_channels, base_features)
        self.enc2 = self.conv_block(base_features, base_features * 2)
        self.enc3 = self.conv_block(base_features * 2, base_features * 4)
        self.enc4 = self.conv_block(base_features * 4, base_features * 8)

        # Bottleneck
        self.bottleneck = self.conv_block(base_features * 8, base_features * 16)

        # Decoder
        self.up4 = self.upconv(base_features * 16, base_features * 8)
        self.dec4 = self.conv_block(base_features * 16, base_features * 8)
        self.up3 = self.upconv(base_features * 8, base_features * 4)
        self.dec3 = self.conv_block(base_features * 8, base_features * 4)
        self.up2 = self.upconv(base_features * 4, base_features * 2)
        self.dec2 = self.conv_block(base_features * 4, base_features * 2)
        self.up1 = self.upconv(base_features * 2, base_features)
        self.dec1 = self.conv_block(base_features * 2, base_features)

        # Final layer
        self.final = nn.Conv2d(base_features, num_classes, kernel_size=1)

    def conv_block(self, cin, cout):
        """Convolutional block with residual connections."""
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cout),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cout),
        )

    def upconv(self, cin, cout):
        """Upsampling block with bilinear interpolation followed by convolution."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(cin, cout, kernel_size=1),
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder path
        d4 = self.up4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Final layer
        return self.final(d1)
