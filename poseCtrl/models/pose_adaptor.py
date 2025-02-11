import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D

class VPmatrixEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=64, output_size=(77, 77)):
        super(VPmatrixEncoder, self).__init__()

        # 输入层：调整通道数
        self.input_layer = nn.Conv2d(input_channels, base_channels, kernel_size=1)

        # ResNet Blocks (4x4 -> 8x8 -> 16x16 -> 32x32)
        self.res_block1 = ResnetBlock2D(base_channels, base_channels * 2)
        self.res_block2 = ResnetBlock2D(base_channels * 2, base_channels * 4)
        self.res_block3 = ResnetBlock2D(base_channels * 4, base_channels * 8)

        # 上采样部分 (32x32 -> 64x64 -> 77x77)
        self.upsample1 = Upsample2D(base_channels * 8)
        self.upsample2 = Upsample2D(base_channels * 4)
        self.upsample3 = nn.ConvTranspose2d(base_channels * 4, 1, kernel_size=3, stride=1, padding=1) 

    def forward(self, x):
        # 调整输入形状
        x = x.unsqueeze(1)  # (batch, 4, 4) -> (batch, 1, 4, 4)
        x = self.input_layer(x)  # (batch, 1, 4, 4) -> (batch, 64, 4, 4)

        # ResNet 特征提取 + 残差连接
        x_res = self.res_block1(x) + x  # (batch, 64, 8, 8)
        x_res = self.res_block2(x_res) + x_res  # (batch, 128, 16, 16)
        x_res = self.res_block3(x_res) + x_res  # (batch, 256, 32, 32)

        # 上采样到 77x77
        x_up = self.upsample1(x_res)  # (batch, 128, 64, 64)
        x_up = self.upsample2(x_up)  # (batch, 64, 128, 128)
        x_up = F.interpolate(x_up, size=(77, 77), mode='bilinear', align_corners=True)  # (batch, 64, 77, 77)
        x_up = self.upsample3(x_up).squeeze(1)  # (batch, 1, 77, 77) -> (batch, 77, 77)

        return x_up

import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from poseCtrl.data.dataset import CustomDataset
path = r"F:\Projects\diffusers\ProgramData\pic"
dataset=CustomDataset(path)
data=dataset[0]
# # 测试
# batch_size = 8
# vp_matrix = torch.randn(batch_size, 4, 4)  # 生成随机 VP 矩阵
vp_matrix=data['projection_matrix']@data['view_matrix']
vp_matrix_tensor = torch.from_numpy(vp_matrix).float().unsqueeze(0)
print(vp_matrix.shape)
model = VPmatrixEncoder()
output = model(vp_matrix_tensor)

print("输入形状:", vp_matrix.shape)  # (8, 4, 4)
print("输出形状:", output.shape)  # (8, 77, 77)
