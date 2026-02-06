""" 
input: image [3,512,512]
        vp_matrix [4,4]
        text_input

basemodel: base smplx model

dataloader: 
        CombinedDatasetTest(new):
point(base_model) encoder:
        PointNetEncoder
PoseAttnProcessor:
        PoseAttnProcessorV7
inference:
        PoseControlNetV7
        colabV4
"""

""" 
DATASET STRUCTURE:

processor = CLIPImageProcessor()
tokenizer = CLIPTokenizer.from_pretrained(r'F:\Projects\diffusers\ProgramData\basemodel', subfolder="tokenizer")
data_root_path_2 = r"F:\Projects\diffusers\ProgramData\test"
txt_subdir_name = r'F:\Projects\diffusers\ProgramData\new_data\image\smpl'
train_dataset = CombinedDatasetTest(
    path2=data_root_path_2,
    tokenizer=tokenizer,
    txt_subdir_name=txt_subdir_name
)

训练过程每一个类别的图片需要有五类文件：
1. 原图：从压缩包解压
2. merged_joints2d.txt: copy进去, 每个类别的joints不一样，注意区分
3. image_features.txt: copy进去， 每个类别的feature是一样的
4. image_smpl.txt: copy进去， 每个类别的smpl关节是不一样的，注意区分，目前只有normal视角的，没有mirror的
5. txt_subdir_name: 传进CombinedDatasetTest，每个类别也不一样，现在只有normal的，目前的dataset也只支持读入normal，没有mirror的

EXAMPLE:
!unzip images/image_resized.zip
!cp images/merged_joints2d.txt image_resized/merged_joints2d.txt
!cp images/filtered_camera_params.txt image_resized/camera_params.txt
!cp images/image_features.txt image_resized/image_features.txt
!cp images/image_smpl_normal.txt image_resized/image_smpl.txt


"""
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPProcessor
import sys
from pathlib import Path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir, "PoseCtrl")
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir,"poseCtrl"))
from poseCtrl.models.pose_adaptor import VPmatrixPoints, ImageProjModel, VPmatrixPointsV1, VPProjModel, PointNetEncoder
from poseCtrl.models.attention_processor import AttnProcessor, PoseAttnProcessorV4, PoseAttnProcessorV7
from poseCtrl.data.dataset import CustomDataset_v4, load_base_points, CombinedDataset, CombinedDatasetTest
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
from poseCtrl.models.posectrl import PoseCtrlV4_val
from diffusers import StableDiffusionPipeline
from poseCtrl.models.pose_controlnet import PoseControlNetModel 

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str, 
        default='/ckpt/basemodel',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_pose_path",
        type=str,
        default=None,
        help="Path to pretrained  posectrl model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--base_point_path1",
        type=str,
        default=r'dataSet/standardVertex.txt',
        help='Path to base model points'
    )
    parser.add_argument(
        "--base_point_path2",
        type=str,
        default=r'dataSet/standardVertex_2.txt',
        help='Path to base model points'
    )
    parser.add_argument(
        "--data_root_path_1",
        type=str,
        default="/images/pic",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--data_root_path_2",
        type=str,
        default="C:\\Users\\31878\\Desktop\\111\\image_resized",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--data_root_path_3",
        type=str,
        default="/images/image_mirror_resized",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--data_root_path_4",
        type=str,
        default="/images/image_left",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--data_root_path_5",
        type=str,
        default="/images/image_right",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--val_data_root_path_2",
        type=str,
        default="/images/image_test",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--txt_subdir_name",
        type=str,
        default=r"C:\Users\31878\Desktop\111\smpl",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--lr_ft",
        type=float,
        default=1e-4,
        help="Learning rate for feature transformation regularizer.",
    )
    parser.add_argument(
        "--lr_balance",
        type=float,
        default=1e-4,
        help="Learning rate for balance regularizer.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="/image_encoder",
        # required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-pose_ctrl",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

from torchvision import transforms

def denormalize(tensor, mean, std):
    return tensor * std + mean

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def validation(pose_model, save_path, val_dataloader, device):
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            image = data['image'][0].to(device)
            vmatrix = data['view_matrix'].to(torch.float32).to(device)
            pmatrix = data['projection_matrix'].to(torch.float32).to(device)
            text = 'highly detailed, anime, 1girl, blue_eyes, long_hair, dress, smile, simple_background'

            images = pose_model.generate(prompt=text, num_samples=4, num_inference_steps=50, seed=42, V_matrix=vmatrix, P_matrix=pmatrix)
            images = image_grid(images, 1, 4)
            save_img_path = os.path.join(save_path, f"image_{idx}.png")
            image = denormalize(image, 0.5, 0.5)
            image_pil = transforms.ToPILImage()(image)

            combined_image = Image.new('RGB', (5*image_pil.width, image_pil.height))
            combined_image.paste(image_pil, (0, 0))
            combined_image.paste(images, (image_pil.width, 0))
            combined_image.save(save_img_path)


from collections import defaultdict
import random
from torch.utils.data import Sampler

class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.type_to_indices = defaultdict(list)

        for idx in range(len(dataset)):
            sample_type = dataset.samples[idx]['type']
            self.type_to_indices[sample_type].append(idx)

        self.batches = []
        for sample_type, indices in self.type_to_indices.items():
            random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) == batch_size:
                    self.batches.append(batch)

        random.shuffle(self.batches)  # 打乱 batch 顺序

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def change_checkpoint(checkpoint, new_checkpoint_path):
    sd = checkpoint
    image_proj_model_point_sd = {}
    atten_sd = {}
    unet_copy_sd = {}
    for k in sd:
        if k.startswith("unet_copy"):
            unet_copy_sd[k.replace("unet_copy.", "")] = sd[k]
        elif k.startswith("unet"):
            pass
        elif k.startswith("image_proj_model_point"):
            image_proj_model_point_sd[k.replace("image_proj_model_point.", "")] = sd[k]
        elif k.startswith("atten_modules"):
            atten_sd[k.replace("atten_modules.", "")] = sd[k]
    new_checkpoint_path = Path(new_checkpoint_path, "posectrl.bin")
    torch.save({"image_proj_model_point": image_proj_model_point_sd, "atten_modules": atten_sd, "unet_copy":unet_copy_sd}, new_checkpoint_path)

import torch

def custom_collate_fn(batch):
    """
    自定义 collate 函数，适配 CombinedDatasetTest.__getitem__ 的输出。
    必选字段（逐元素均存在并能直接 stack）：
      - image: [3, 512, 512] float
      - projection_matrix: [4, 4] float
      - view_matrix: [4, 4] float
      - text: str
      - type: str
      - text_input_ids: [1, L] long
      - image_path: str

    可选字段（可能缺失）：
      - feature: [3, 512, 512] float（仅 v1）
      - joints_image: [3, 512, 512] float（仅 v4）
      - param_smpl: [21, 3] float（仅 v4，可为 None）
      - points: [N, 3] float（仅 v4，可为 None，N 变长）

    返回：
      - 对应张量堆叠/填充后的 batch 字典
    """

    # 1) 必选字段：直接堆叠
    image_tensors = torch.stack([d['image'] for d in batch])                          # [B, 3, 512, 512]
    proj_matrices = torch.stack([d['projection_matrix'] for d in batch])              # [B, 4, 4]
    view_matrices = torch.stack([d['view_matrix'] for d in batch])                    # [B, 4, 4]
    texts = [d['text'] for d in batch]                                                # list[str]
    types = [d['type'] for d in batch]                                                # list[str]
    image_paths = [d.get('image_path', '') for d in batch]                             # list[str]

    # text_input_ids: 每样本形状 [1, L] -> 压成 [L] 再 stack 成 [B, L]
    if any('text_input_ids' not in d for d in batch):
        raise KeyError("Missing 'text_input_ids' in some samples. Ensure dataset returns it.")
    text_ids_list = [d['text_input_ids'].squeeze(0) for d in batch]
    # 校验序列长度一致（DataLoader 层面通常已 pad 到 tokenizer.model_max_length）
    seq_lens = {t.size(0) for t in text_ids_list}
    if len(seq_lens) != 1:
        raise ValueError(f"text_input_ids length mismatch across batch: {seq_lens}")
    text_input_ids = torch.stack(text_ids_list, dim=0)                                # [B, L]

    collated_batch = {
        'image': image_tensors,
        'projection_matrix': proj_matrices,
        'view_matrix': view_matrices,
        'text': texts,
        'type': types,
        'text_input_ids': text_input_ids,
        'image_path': image_paths,
    }

    # 2) 可选字段：feature（v1）
    if any('feature' in d for d in batch):
        feature_sample = next(d['feature'] for d in batch if 'feature' in d)
        zero_feature_placeholder = torch.zeros_like(feature_sample)
        feature_tensors = [d.get('feature', zero_feature_placeholder) for d in batch]
        collated_batch['feature'] = torch.stack(feature_tensors)                       # [B, 3, 512, 512]
        collated_batch['has_feature'] = torch.tensor(
            [int('feature' in d) for d in batch], dtype=torch.uint8
        )                                                                              # [B]

    # 3) 可选字段：joints_image（v4）
    if any('joints_image' in d for d in batch):
        joints_sample = next(d['joints_image'] for d in batch if 'joints_image' in d)
        blank_joint = torch.zeros_like(joints_sample)
        joints_images = [d.get('joints_image', blank_joint) for d in batch]
        collated_batch['joints_image'] = torch.stack(joints_images)                    # [B, 3, 512, 512]
        collated_batch['has_joints_image'] = torch.tensor(
            [int('joints_image' in d) for d in batch], dtype=torch.uint8
        )                                                                              # [B]

    # 4) 可选字段：param_smpl（固定 21×3；v4）
    # 存在则 stack；缺失用零占位；同时给出 mask
    has_param = []
    param_tensors = []
    for d in batch:
        ps = d.get('param_smpl', None)
        if ps is None:
            has_param.append(0)
            param_tensors.append(torch.zeros(21, 3, dtype=torch.float32))
        else:
            # 若形状不匹配则抛错，避免静默错误
            if ps.ndim != 2 or ps.size(0) != 21 or ps.size(1) != 3:
                raise ValueError(f"param_smpl must be [21,3], got {tuple(ps.size())}")
            has_param.append(1)
            param_tensors.append(ps)
    # 只有当 batch 中至少有一个样本包含 param_smpl，才返回该字段
    if any(has_param):
        collated_batch['param_smpl'] = torch.stack(param_tensors)                      # [B, 21, 3]
        collated_batch['has_param_smpl'] = torch.tensor(has_param, dtype=torch.uint8) # [B]

    # 5) 可选字段：points（变长 N×3；v4）
    # 采用按 batch 的零填充，并返回 lengths 和 mask
    points_list = [d.get('points', None) for d in batch]
    if any(p is not None for p in points_list):
        lengths = [p.size(0) if p is not None else 0 for p in points_list]
        max_len = max(lengths)
        if max_len == 0:
            # 逻辑上不会发生（前面 any 为 True），但保底
            pass
        else:
            padded = torch.zeros(len(batch), max_len, 3, dtype=torch.float32)
            mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
            for i, p in enumerate(points_list):
                if p is None:
                    continue
                # 形状校验
                if p.ndim != 2 or p.size(1) != 3:
                    raise ValueError(f"points must be [N,3], got {tuple(p.size())}")
                n = p.size(0)
                padded[i, :n, :] = p
                mask[i, :n] = True
            collated_batch['points'] = padded                                          # [B, max_N, 3]
            collated_batch['points_mask'] = mask                                       # [B, max_N] (True=有效)
            collated_batch['points_lengths'] = torch.tensor(lengths, dtype=torch.int32)# [B]
            collated_batch['has_points'] = (collated_batch['points_lengths'] > 0).to(torch.uint8)

    return collated_batch



class posectrl(nn.Module):
    def __init__(self, unet, unet_copy, pointnet_encoder, atten_modules_p, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.unet_copy = unet_copy
        self.image_proj_model_point = pointnet_encoder
        self.atten_modules = atten_modules_p

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, point_embeds, V_matrix, P_matrix):
        point_tokens, trans_feat, importance = self.image_proj_model_point(point_embeds, V_matrix, P_matrix, encoder_hidden_states)

        point_hidden_states = torch.cat([encoder_hidden_states, point_tokens], dim = 1)

        down_block_res_samples, mid_block_res_sample = self.unet_copy(
                    noisy_latents,
                    timesteps,
                    point_hidden_states,
                    return_dict=False,
                )
        noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
        
        return noise_pred, trans_feat, importance

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_VPmatrix_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model_point.parameters()]))
        orig_atten_sum = torch.sum(torch.stack([torch.sum(p) for p in self.atten_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model_point.load_state_dict(state_dict["image_proj_model_point"], strict=True)
        self.atten_modules.load_state_dict(state_dict["atten_modules"], strict=True)
        self.unet_copy.load_state_dict(state_dict['unet_copy'],strict=True)

        # Calculate new checksums
        new_VPmatrix_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model_point.parameters()]))
        new_atten_sum = torch.sum(torch.stack([torch.sum(p) for p in self.atten_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_VPmatrix_sum != new_VPmatrix_sum, "Weights of VPmatrixEncoder did not change!"
        assert orig_atten_sum != new_atten_sum, "Weights of atten_modules did not change!"
        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    unet_copy = PoseControlNetModel.from_unet(unet)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    unet_copy.requires_grad_(True)
    
    # for name, processor in unet_copy.attn_processors.items():
    #     for param_name, param in processor.named_parameters():
    #         if "to_k" in param_name or "to_v" in param_name:
    #             param.requires_grad = False
    
    raw_base_points1=load_base_points(args.base_point_path1) 
    vpmatrix_points_sd1 = VPmatrixPointsV1(raw_base_points1)
    pointnet_encoder = PointNetEncoder(channel=3).to("cuda")

    # init pose modules
    attn_procs = {}
    unet_sd = unet_copy.state_dict()
    for name in unet_copy.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet_copy.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet_copy.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet_copy.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet_copy.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_pose.weight": unet_sd[layer_name + ".to_k.weight"].clone(),
                "to_v_pose.weight": unet_sd[layer_name + ".to_v.weight"].clone(),
            }
            attn_procs[name] = PoseAttnProcessorV7(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)

    unet_copy.set_attn_processor(attn_procs)

    atten_modules = torch.nn.ModuleList(unet_copy.attn_processors.values())
    atten_modules.requires_grad_(True)
    pose_ctrl = posectrl(unet, unet_copy, pointnet_encoder, atten_modules, args.pretrained_pose_path)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = itertools.chain(pose_ctrl.image_proj_model_point.parameters(),  pose_ctrl.atten_modules.parameters(), pose_ctrl.unet_copy.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    # train_dataset = CustomDataset_v4(args.data_root_path, camera_params_file=args.CAMERA_PARAMS_FILE, image_features_file=args.IMAGE_FEATURES_FILE)
    train_dataset = CombinedDatasetTest(
        # path1=args.data_root_path_1,
        path2=args.data_root_path_2,
        # path3=args.data_root_path_3,
        # path4=args.data_root_path_4,
        # path5=args.data_root_path_5,
        tokenizer=tokenizer,
        txt_subdir_name=args.txt_subdir_name
    )

    # Test dataset loading
    print(f"Dataset length: {len(train_dataset)}")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("Sample keys:", list(sample.keys()))
        if 'points' in sample:
            print("Points shape:", sample['points'].shape if sample['points'] is not None else "None")
        else:
            print("No 'points' key in sample")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=GroupedBatchSampler(train_dataset, batch_size=args.train_batch_size),
        collate_fn=custom_collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # val_dataset = CombinedDataset(
    #     # path1=args.data_root_path_1,
    #     path2=args.val_data_root_path_2,
    # )

    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_sampler=GroupedBatchSampler(val_dataset, batch_size=1),
    #     collate_fn=custom_collate_fn,
    #     num_workers=args.dataloader_num_workers,
    # )

    # Prepare everything with our `accelerator`.
    pose_ctrl, optimizer, train_dataloader = accelerator.prepare(pose_ctrl, optimizer, train_dataloader)

    global_step = 0
    for epoch in range(0, args.num_train_epochs): #default is 100
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(pose_ctrl):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["image"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                min_step = 10
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():  
                    if batch['type'][0]=='v1':
                        base_points = vpmatrix_points_sd1(batch['view_matrix'], batch['projection_matrix'])
                        image_tensor = processor(images=base_points, return_tensors="pt",do_rescale=False).pixel_values
                        point_embeds = image_encoder(image_tensor.to(accelerator.device, dtype=weight_dtype)).image_embeds
                    elif batch['type'][0]=='v4':
                        # image_tensor = processor(images=batch['joints_image'], return_tensors="pt", do_rescale=False).pixel_values
                        # point_embeds = image_encoder(image_tensor.to(accelerator.device, dtype=weight_dtype)).image_embeds
                        if batch['points'] is None:
                            print("Error: batch['points'] is None for v4 type")
                            raise ValueError("Points not found")
                        point_embeds = batch['points'].to(accelerator.device, dtype=weight_dtype)

                if "text_input_ids" in batch:
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                else:
                    text_input_ids = tokenizer(
                            "a highly detailed anime girl, in front of a pure black background",
                            max_length=tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        ).input_ids
                    encoder_hidden_states = text_encoder(text_input_ids.to(accelerator.device))[0]
                    encoder_hidden_states = encoder_hidden_states.repeat(args.train_batch_size, 1, 1) 
                
                noise_pred, trans_feat, importance = pose_ctrl(noisy_latents, timesteps, encoder_hidden_states, point_embeds, batch['view_matrix'], batch['projection_matrix'])
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                ft_loss = feature_transform_reguliarzer(trans_feat)
                # balance_loss = torch.std(importance)
                balance_loss = torch.sqrt(torch.var(importance) + 1e-8)
                loss = loss + args.lr_ft * ft_loss + args.lr_balance * balance_loss
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                change_checkpoint(pose_ctrl.state_dict(), save_path)

            begin = time.perf_counter()


if __name__ == "__main__":
    main()  


