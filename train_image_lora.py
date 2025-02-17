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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from poseCtrl.models.pose_adaptor import VPmatrixPoints, ImageProjModel
from poseCtrl.models.attention_processor import AttnProcessor, PoseAttnProcessor
from poseCtrl.data.dataset import CustomDataset, load_base_points

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str, 
        default='runwayml/stable-diffusion-v1-5',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_pose_path",
        type=str,
        default=None,
        help="Path to pretrained  posectrl model. If not specified weights are initialized randomly.",
    )
    # parser.add_argument(
    #     "--data_json_file",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Training data",
    # )
    parser.add_argument(
        "--base_point_path",
        type=str,
        default=r'F:\Projects\diffusers\Project\PoseCtrl\dataSet\standardVertex.txt',
        help='Path to base model points'
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="F:\\Projects\\diffusers\\ProgramData\\sample_new",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
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
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
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
        default=2,
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

class posectrl(nn.Module):
    def __init__(self, unet, vpmatrix_points, image_proj_model, atten_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.vpmatrix_points = vpmatrix_points
        self.atten_modules = atten_modules

        self.image_proj_model = image_proj_model

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, V_matrix, P_matrix, image_embeds):
        point_tokens = self.vpmatrix_points(V_matrix, P_matrix)
        feature_tokens = self.image_proj_model(image_embeds)
        """ 修改:防止之后要加text """
        if encoder_hidden_states:
            encoder_hidden_states = torch.cat([point_tokens, feature_tokens, encoder_hidden_states], dim=1)
        else:
            encoder_hidden_states=torch.cat([point_tokens, feature_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_VPmatrix_sum = torch.sum(torch.stack([torch.sum(p) for p in self.vpmatrix_points.parameters()]))
        orig_atten_sum = torch.sum(torch.stack([torch.sum(p) for p in self.atten_modules.parameters()]))
        orig__proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.vpmatrix_points.load_state_dict(state_dict["vpmatrix_points"], strict=True)
        self.atten_modules.load_state_dict(state_dict["atten_modules"], strict=True)
        self.image_proj_model.load_state_dict(state_dict["image_proj_model"], strict=True)

        # Calculate new checksums
        new_VPmatrix_sum = torch.sum(torch.stack([torch.sum(p) for p in self.vpmatrix_points.parameters()]))
        new_atten_sum = torch.sum(torch.stack([torch.sum(p) for p in self.atten_modules.parameters()]))
        new__proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))

        # Verify if the weights have changed
        assert orig_VPmatrix_sum != new_VPmatrix_sum, "Weights of VPmatrixEncoder did not change!"
        assert orig_atten_sum != new_atten_sum, "Weights of atten_modules did not change!"
        assert orig__proj_sum != new__proj_sum, "Weights of image_proj_model did not change!"
        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

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
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    #vp-matrix encoder
    raw_base_points=load_base_points(args.base_point_path)  
    vpmatrix_points_sd = VPmatrixPoints(raw_base_points)
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    # init pose modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = PoseAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)

    unet.set_attn_processor(attn_procs)

    atten_modules = torch.nn.ModuleList(unet.attn_processors.values())
    atten_modules.requires_grad_(True)
    print(atten_modules)
    pose_ctrl = posectrl(unet, vpmatrix_points_sd, image_proj_model, atten_modules, args.pretrained_pose_path)
    print(pose_ctrl.atten_modules.state_dict().keys())  # 这里应该有内容

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weiaght_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(pose_ctrl.vpmatrix_points.parameters(),  pose_ctrl.atten_modules.parameters(), pose_ctrl.image_proj_model.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    


if __name__ == "__main__":
    main()  