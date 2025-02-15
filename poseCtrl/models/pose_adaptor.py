import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D, Upsample2D
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from poseCtrl.data.dataset import load_base_points


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


import torch
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D

class VPmatrixPoints(nn.Module):
    """ 
    Input:  
        V_matrix: [batch,4,4]
        P_matrix: [batch,4,4]
        raw_base_points: [13860,4]
    Output:
        base_points: [batch,77,768]
    """
    def __init__(self, raw_base_points):
        super().__init__() 
        self.register_buffer("raw_base_points", raw_base_points)

        self.resnet = nn.ModuleList([
            nn.Conv2d(720, 256, kernel_size=(3, 3), padding=(1, 1)),  
            ResnetBlock2D(in_channels=256, out_channels=256, temb_channels=None),  
            ResnetBlock2D(in_channels=256, out_channels=512, temb_channels=None),  
            ResnetBlock2D(in_channels=512, out_channels=768, temb_channels=None),  
            nn.Conv2d(768, 768, kernel_size=(1, 1))  
        ])

    def forward(self, V_matrix, P_matrix):
        VP_matrix = torch.bmm(P_matrix, V_matrix)  # [batch, 4, 4]
        points = self.raw_base_points.unsqueeze(0).expand(VP_matrix.shape[0], -1, -1)
        transformed_points = torch.bmm(points, VP_matrix.transpose(1, 2))  # [batch, 13860, 4]

        base_points = transformed_points.view(VP_matrix.shape[0], 77, 720)
        base_points = base_points.permute(0, 2, 1).unsqueeze(-1)  # [batch, 720, 77] → [batch, 720, 77, 1]

        for layer in self.resnet:
            if isinstance(layer, ResnetBlock2D):
                base_points = layer(base_points, temb=None)  
            else:
                base_points = layer(base_points)

        base_points = base_points.squeeze(-1).permute(0, 2, 1)  # [batch, 77, 768]

        return base_points


        

# --------------------- Dataset & Testing ---------------------

# import numpy as np

# from poseCtrl.data.dataset import CustomDataset

# path = r"F:\\Projects\\diffusers\\ProgramData\\sample"
# dataset = CustomDataset(path)
# data = dataset[0]

# # Generate VP Matrix
# vp_matrix = data['projection_matrix'] @ data['view_matrix']
# model = VPmatrixEncoder()
# vp_matrix_tensor = vp_matrix.float().unsqueeze(0)

# # Model Testing
# model = VPmatrixEncoder()
# output = model(vp_matrix_tensor)

# print("Input shape:", vp_matrix_tensor.shape)  # Expected: (1, 1, 4, 4)
# print("Output shape:", output.shape)  # Expected: (1, 77, 77)


# path=r'F:\Projects\diffusers\Project\PoseCtrl\dataSet\standardVertex.txt'
# raw_base_points=load_base_points(path)
# points = VPmatrixPoints(raw_base_points)
# with torch.no_grad():
#     base_points=points(data['view_matrix'].unsqueeze(0), data['projection_matrix'].unsqueeze(0))
# print(base_points.shape)