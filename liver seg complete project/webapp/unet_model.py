import torch
import torch.nn as nn
import torch.nn.functional as F

# EXACT same UNet as in your training script
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
        ])
        self.decoder = nn.ModuleList([
            self.conv_block(1024, 512),
            self.conv_block(512, 256),
            self.conv_block(256, 128),
            self.conv_block(128, 64),
        ])
        self.final = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i](x)

        return self.final(x)
