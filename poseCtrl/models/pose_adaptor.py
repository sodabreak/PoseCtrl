import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D

class VPmatrixEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=64, output_size=(77, 77)):
        super(VPmatrixEncoder, self).__init__()

        # Input Layer
        self.input_layer = nn.Conv2d(input_channels, base_channels, kernel_size=1)

        # ResNet Blocks (Ensure `temb_channels=None` is passed)
        self.res_block1 = ResnetBlock2D(
            in_channels=base_channels, out_channels=base_channels * 2, temb_channels=None
        )
        self.res_block2 = ResnetBlock2D(
            in_channels=base_channels * 2, out_channels=base_channels * 4, temb_channels=None
        )
        self.res_block3 = ResnetBlock2D(
            in_channels=base_channels * 4, out_channels=base_channels * 8, temb_channels=None
        )

        # Upsampling
        self.upsample1 = Upsample2D(channels=base_channels * 8)  # Output: base_channels * 8
        self.upsample2 = Upsample2D(channels=base_channels * 4)  # Output: base_channels * 4
        self.final_conv = nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Ensure input has correct shape (batch, 1, 4, 4)
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.input_layer(x)

        # ResNet Feature Extraction (Passing `temb=None`)
        x = self.res_block1(x, temb=None)
        x = self.res_block2(x, temb=None)
        x = self.res_block3(x, temb=None)

        # Upsample Step (Ensure matching channels)
        if x.shape[1] != self.upsample1.channels:
            x = nn.Conv2d(x.shape[1], self.upsample1.channels, kernel_size=1)(x)

        x = self.upsample1(x)  # Expected Output: (batch, base_channels * 8, 8, 8)

        if x.shape[1] != self.upsample2.channels:
            x = nn.Conv2d(x.shape[1], self.upsample2.channels, kernel_size=1)(x)

        x = self.upsample2(x)  # Expected Output: (batch, base_channels * 4, 16, 16)

        # Interpolation to 77x77
        x = F.interpolate(x, size=(77, 77), mode='bilinear', align_corners=True)
        
        # Final Convolution to Ensure Output Shape
        x = self.final_conv(x).squeeze(1)  # (batch, 1, 77, 77) -> (batch, 77, 77)

        return x

 

# --------------------- Dataset & Testing ---------------------

import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from poseCtrl.data.dataset import CustomDataset

path = r"F:\Projects\diffusers\ProgramData\pic"
dataset = CustomDataset(path)
data = dataset[0]

# Generate VP Matrix
vp_matrix = data['projection_matrix'] @ data['view_matrix']
model = VPmatrixEncoder()
vp_matrix_tensor = torch.from_numpy(vp_matrix).float().unsqueeze(0)

# Model Testing
model = VPmatrixEncoder()
output = model(vp_matrix_tensor)

print("Input shape:", vp_matrix_tensor.shape)  # Expected: (1, 1, 4, 4)
print("Output shape:", output.shape)  # Expected: (1, 77, 77)

