import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D, Upsample2D
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from poseCtrl.data.dataset import load_base_points
import cv2
import numpy as np


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
        transformed_points[..., :3] = torch.where(
            transformed_points[..., 3:4] != 0,
            transformed_points[..., :3] / transformed_points[..., 3:4],
            transformed_points[..., :3]  
        ) # [batch, 13860, 3]
        transformed_points = transformed_points[..., :3]
        ones = torch.ones_like(transformed_points[..., :1])  # Create a tensor of ones with shape [batch, 13860, 1]
        transformed_points = torch.cat([transformed_points, ones], dim=-1)
        base_points = transformed_points.view(VP_matrix.shape[0], 77, 720)
        base_points = base_points.permute(0, 2, 1).unsqueeze(-1)  # [batch, 720, 77] → [batch, 720, 77, 1]

        for layer in self.resnet:
            if isinstance(layer, ResnetBlock2D):
                base_points = layer(base_points, temb=None)  
            else:
                base_points = layer(base_points)

        base_points = base_points.squeeze(-1).permute(0, 2, 1)  # [batch, 77, 768]

        return base_points


class VPmatrixPointsV3(nn.Module):
    """ 
    Input:  
        V_matrix: [batch,4,4]
        P_matrix: [batch,4,4]
        raw_base_points: [13860,4]
    Output:
        base_points: [batch,4,768] 
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
        
        # 新增部分：将77映射到4，并加ReLU
        self.linear = nn.Linear(77, 4)
        self.relu = nn.ReLU()

    def forward(self, V_matrix, P_matrix):
        VP_matrix = torch.bmm(P_matrix, V_matrix)  # [batch, 4, 4]
        points = self.raw_base_points.unsqueeze(0).expand(VP_matrix.shape[0], -1, -1)
        transformed_points = torch.bmm(points, VP_matrix.transpose(1, 2))  # [batch, 13860, 4]
        transformed_points[..., :3] = torch.where(
            transformed_points[..., 3:4] != 0,
            transformed_points[..., :3] / transformed_points[..., 3:4],
            transformed_points[..., :3]
        )  # [batch, 13860, 3]
        transformed_points = transformed_points[..., :3]
        ones = torch.ones_like(transformed_points[..., :1])  # [batch, 13860, 1]
        transformed_points = torch.cat([transformed_points, ones], dim=-1)
        base_points = transformed_points.view(VP_matrix.shape[0], 77, 720)
        base_points = base_points.permute(0, 2, 1).unsqueeze(-1)  # [batch, 720, 77, 1]

        for layer in self.resnet:
            if isinstance(layer, ResnetBlock2D):
                base_points = layer(base_points, temb=None)  
            else:
                base_points = layer(base_points)

        base_points = base_points.squeeze(-1).permute(0, 2, 1)  # [batch, 77, 768]
        base_points = base_points.permute(0, 2, 1)              # [batch, 768, 77]
        base_points = self.linear(base_points)                  # [batch, 768, 4]
        base_points = self.relu(base_points)                    # ReLU激活
        base_points = base_points.permute(0, 2, 1)              # [batch, 4, 768]

        return base_points


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    
class VPProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.vp_linear = torch.nn.Linear(4 * 4, 768)
        self.activation = nn.Sigmoid()

    def forward(self, image_embeds, V_matrix, P_matrix):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)

        VP_matrix = torch.bmm(P_matrix, V_matrix)
        VP_embeds = self.vp_linear(VP_matrix.view(VP_matrix.shape[0], -1))
        VP_embeds = self.activation(VP_embeds)
        VP_embeds = VP_embeds.unsqueeze(1).repeat(1, self.clip_extra_context_tokens, 1)
        return clip_extra_context_tokens + VP_embeds

class VPmatrixPointsV1(nn.Module):
    """ 
    Input:  
        V_matrix: [batch,4,4]
        P_matrix: [batch,4,4]
        raw_base_points: [13860,4]
    Output:
        base_points: image
    """
    def __init__(self, raw_base_points,image_width = 512,image_height=512):
        super().__init__() 
        self.register_buffer("raw_base_points", raw_base_points)
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, V_matrix, P_matrix):
        VP_matrix = torch.bmm(P_matrix, V_matrix)  # [batch, 4, 4]
        points = self.raw_base_points.unsqueeze(0).expand(VP_matrix.shape[0], -1, -1).to('cuda')
        transformed_points = torch.bmm(points, VP_matrix.transpose(1, 2))  # [batch, 13860, 4]
        transformed_points[..., :3] = torch.where(
            transformed_points[..., 3:4] != 0,
            transformed_points[..., :3] / transformed_points[..., 3:4],
            transformed_points[..., :3]  
        ) # [batch, 13860, 3]
        transformed_points = transformed_points[..., :3]
        image_width, image_height = self.image_width, self.image_height

        screen_coords = transformed_points.clone()
        screen_coords[..., 0] = (screen_coords[..., 0] + 1) * 0.5 * image_width   # X: [-1,1] -> [0,512]
        screen_coords[..., 1] = (1 - (screen_coords[..., 1] + 1) * 0.5) * image_height  # Y 翻转: [-1,1] -> [512,0]

        screen_coords = screen_coords.round().long()  # [batch, 13860, 3]

        batch_size = screen_coords.shape[0]
        tensor_images = torch.zeros((batch_size, 3, image_height, image_width), dtype=torch.uint8)

        for b in range(batch_size):
            pixels = screen_coords[b].cpu().numpy()
            image_array = np.full((image_height, image_width), 255, dtype=np.uint8)

            for x, y, _ in pixels:
                if 0 <= x < image_width and 0 <= y < image_height:
                    image_array[y, x] = 0  
            inverted_array = 255 - image_array
            kernel = np.ones((9, 9), np.uint8)  
            dilated_image = cv2.dilate(inverted_array, kernel, iterations=1)  
            smoothed_image = cv2.GaussianBlur(dilated_image, (9,9), 0)
            _, binary_mask = cv2.threshold(smoothed_image, 100, 255, cv2.THRESH_BINARY)
            binary_mask_3ch = np.stack([binary_mask] * 3, axis=-1)  # [512, 512, 3]
            tensor_images[b] = torch.from_numpy(binary_mask_3ch).permute(2, 0, 1)
        return tensor_images.float() / 255   
    
def create_depth_map(screen_coords, transformed_points_ndc, image_height, image_width, 
                     smooth=True, kernel_size=4, iterations=5):
    """
    最终版 V9: 使用 F.max_pool2d 的技巧实现形态学腐蚀（Min Pooling）。
    - 确保“近大远小”，前景深度均匀分布在 [0.2, 0.8]，背景固定为 1.0。
    - smooth=True 时，将前景点的深度值向周围传播，填充轮廓内的空洞。
    """
    batch_size = screen_coords.shape[0]
    
    # --- 步骤 1: 生成稀疏深度图 (逻辑不变) ---
    final_depth_maps = torch.full(
        (batch_size, image_height, image_width), 
        1.0, 
        device=screen_coords.device, 
        dtype=transformed_points_ndc.dtype
    )

    z_buffer = torch.full_like(final_depth_maps, 2.0)

    for b in range(batch_size):
        coords_b = screen_coords[b]
        ndc_depth_b = transformed_points_ndc[b, ..., 2]

        valid_mask = (coords_b[..., 0] >= 0) & (coords_b[..., 0] < image_width) & \
                     (coords_b[..., 1] >= 0) & (coords_b[..., 1] < image_height)
        
        if not valid_mask.any():
            continue

        valid_coords = coords_b[valid_mask].long()
        valid_ndc_depth = ndc_depth_b[valid_mask]
        
        sorted_indices = torch.argsort(valid_ndc_depth, descending=True)
        sorted_coords = valid_coords[sorted_indices]
        sorted_ndc_depth = valid_ndc_depth[sorted_indices]
        z_buffer[b, sorted_coords[:, 1], sorted_coords[:, 0]] = sorted_ndc_depth

    for b in range(batch_size):
        foreground_mask = z_buffer[b] < 2.0
        
        if not foreground_mask.any():
            continue

        foreground_depths_ndc = z_buffer[b][foreground_mask]
        
        ranks_asc = torch.argsort(torch.argsort(foreground_depths_ndc)).float()
        
        num_foreground_pixels = ranks_asc.shape[0]
        if num_foreground_pixels > 1:
            equalized_depths = ranks_asc / (num_foreground_pixels - 1)
            reversed_depths = 1.0 - equalized_depths
            scaled_depths = reversed_depths * 0.6 + 0.2
            final_depth_maps[b][foreground_mask] = scaled_depths.to(final_depth_maps.dtype)
        else:
            final_depth_maps[b][foreground_mask] = torch.tensor(0.5, dtype=final_depth_maps.dtype)

    # --- 步骤 2: 修正后的形态学平滑 ---
    if smooth:
        smoothed_map = final_depth_maps.unsqueeze(1)
        padding = kernel_size // 2
        
        for _ in range(iterations):
            # 关键修正：使用 -max_pool(-x) 来模拟 min_pool(x)
            # 1. 对输入取反。现在前景值（原来是小的）变成大的负数。
            # 2. 手动 pad，用一个非常小的值（-1.0，因为原背景是1.0）
            padded_map = F.pad(-smoothed_map, (padding, padding, padding, padding), mode='constant', value=-1.0)
            
            # 3. 执行 max_pool。这会找到邻域内最大的负数（即原始值最小的数）。
            max_pooled = F.max_pool2d(padded_map, kernel_size=kernel_size, stride=1)
            
            # 4. 再次取反，将结果恢复到原始范围。
            smoothed_map = -max_pooled
            
        return smoothed_map
    else:
        return final_depth_maps.unsqueeze(1)

class VPmatrixPointsDepth(nn.Module):
    """ 
    Input:  
        V_matrix: [batch,4,4]
        P_matrix: [batch,4,4]
        raw_base_points: [13860,4]
    Output:
        depth_map: [batch, 1, H, W]
    """
    def __init__(self, raw_base_points, image_width=512, image_height=512):
        super().__init__() 
        self.register_buffer("raw_base_points", raw_base_points)
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, V_matrix, P_matrix):
        VP_matrix = torch.bmm(P_matrix, V_matrix)  # [batch, 4, 4]
        points = self.raw_base_points.unsqueeze(0).expand(VP_matrix.shape[0], -1, -1).to(V_matrix.device)
        
        # 变换到裁剪空间
        transformed_points_homogeneous = torch.bmm(points, VP_matrix.transpose(1, 2))  # [batch, 13860, 4]
        
        # 透视除法，得到 NDC 坐标 [-1, 1]
        w = transformed_points_homogeneous[..., 3:4]
        transformed_points_ndc = torch.where(
            w != 0,
            transformed_points_homogeneous[..., :3] / w,
            transformed_points_homogeneous[..., :3]
        ) # [batch, 13860, 3]

        # --- 从这里开始修改 ---
        
        # 1. 计算屏幕坐标
        screen_coords = transformed_points_ndc.clone()
        screen_coords[..., 0] = (screen_coords[..., 0] + 1) * 0.5 * self.image_width
        screen_coords[..., 1] = (1 - (screen_coords[..., 1] + 1) * 0.5) * self.image_height
        screen_coords = screen_coords.round().long()

        # 2. 调用新函数生成深度图
        # 我们需要传入 screen_coords (用于位置) 和 transformed_points_ndc (用于深度)
        depth_map = create_depth_map(
            screen_coords, 
            transformed_points_ndc, 
            self.image_height, 
            self.image_width
        )
        
        return depth_map

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
class MoEEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_experts,
                 top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控
        self.gate = nn.Linear(input_dim, num_experts)

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x, point_embeds):
        """
        x: [B, T, D]  ->  out: [B, T, D]: text
        point_embeds: point features
        同时返回负载均衡统计量
        """
        B, T, D = x.shape
        flat = x.reshape(B * T, D)                      # [B*T, D]

        gate_logits = self.gate(flat)                   # [B*T, num_experts]

        gate_probs = F.softmax(gate_logits, dim=-1)     # [B*T, num_experts]
        top_val, top_idx = torch.topk(gate_probs, self.top_k, dim=-1)  # [B*T, top_k]

        importance = torch.zeros(self.num_experts, device=x.device, dtype=top_val.dtype)
        importance.index_add_(0, top_idx.view(-1), top_val.view(-1))

        out = torch.zeros_like(flat)

        D = flat.size(-1)
        expert_1_accumulated_output = torch.zeros_like(flat)
        expert_2_accumulated_output = torch.zeros_like(flat)

        pointfeature = point_embeds

        for i in range(B * T):
            input_i = flat[i] # 当前的输入向量 [D]

            for k in range(self.top_k):
                expert_id = top_idx[i, k].item()
                weight = top_val[i, k]
                expert_output = self.experts[expert_id](input_i) # [D]
                out[i] += weight * expert_output
                if expert_id == 0:
                    expert_1_accumulated_output[i] += weight * expert_output
                elif expert_id == 1:
                    expert_2_accumulated_output[i] += weight * expert_output

        final_point = expert_1_accumulated_output.view(B, T, -1) + pointfeature
        final_out=torch.cat([final_point, expert_2_accumulated_output.view(B, T, -1)], dim=2)


        return importance, final_out

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(1024)
        self.ln4 = nn.LayerNorm(512)
        self.ln5 = nn.LayerNorm(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.ln1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.ln2(self.conv2(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.ln3(self.conv3(x).transpose(1, 2)).transpose(1, 2))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.ln4(self.fc1(x)))
        x = F.relu(self.ln5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
class PointNetEncoder(nn.Module):
    def __init__(self, channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(1024)
        self.fstn = STNkd(k=64)

        self.feature_proj = nn.Linear(10475, 768)
        self.seq_proj = nn.Linear(1024, 77)
        self.norm = nn.LayerNorm(768)
        self.act = nn.GELU()
        self.moe=MoEEncoder(input_dim=768, hidden_dim=512, output_dim=768, num_experts=2, top_k=2)
        self.out=nn.Linear(768*2, 768)

    def forward(self, x, V_matrix, P_matrix, text_feature):
        B, D, N = x.size()
        trans = torch.bmm(P_matrix, V_matrix) 
        new_dim = torch.ones(B, D, 1, device=x.device, dtype=x.dtype)
        x = torch.cat([x, new_dim], dim=2)
        x = torch.bmm(x, trans.transpose(1, 2))
        x[..., :3] = torch.where(
            x[..., 3:4] != 0,
            x[..., :3] / x[..., 3:4],
            x[..., :3]
        )  # [batch, 13860, 3]
        x = x[..., :3]
        x = x.transpose(2, 1)
        x = F.relu(self.ln1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        trans_feat = self.fstn(x).to(x.dtype)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        pointfeat = x

        x = F.relu(self.ln2(self.conv2(x).transpose(1, 2)).transpose(1, 2))
        x = self.ln3(self.conv3(x).transpose(1, 2)).transpose(1, 2)
        x = self.feature_proj(x)      # [b, 1024, 768]
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)         # [b, 768, 1024]
        x = self.seq_proj(x)          # [b, 768, 77]
        x = x.transpose(1, 2)

        importance, x = self.moe(text_feature, x) 
    
        x = self.out(x)
        return x, trans_feat, importance

# --------------------- Dataset & Testing ---------------------

# import numpy as np

# from poseCtrl.data.dataset import CustomDataset

# path = r"F:\\Projects\\diffusers\\ProgramData\\sample_new"
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