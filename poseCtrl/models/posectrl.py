import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from poseCtrl.models.attention_processor import AttnProcessor, CNAttnProcessor, PoseAttnProcessor
# from poseCtrl.models.pose_adaptor import VPmatrixEncoder
from poseCtrl.models.utils import get_generator
import sys
sys.path.append('/content/drive/MyDrive/PoseCtrl')
sys.path.append('/content/drive/MyDrive/PoseCtrl/poseCtrl')
from poseCtrl.models.attention_processor import AttnProcessor, PoseAttnProcessor
from poseCtrl.models.pose_adaptor import VPmatrixPoints, ImageProjModel
from poseCtrl.data.dataset import CustomDataset, load_base_points

class PoseCtrl:
    """ 修改: 输入要加上self.VP, self.BasePoints"""
    def __init__(self, sd_pipe, image_encoder_path, pose_ckpt, raw_base_points, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.pose_ckpt = pose_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_posectrl()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.vpmatrix_points = self.init_VP()
        self.load_image_proj_model = self.init_proj()
        self.load_posectrl()

    def init_VP(self):
        vpmatrix_points = VPmatrixPoints(raw_base_points).to(self.device, dtype=torch.float16)
        return vpmatrix_points

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_posectrl(self):
        unet = self.pipe.unet
        attn_procs = {}
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
                attn_procs[name] = PoseAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_posectrl(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"vpmatrix_points": {}, "atten_modules": {}, "image_proj_model": {}}
            with safe_open(self.pose_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("vpmatrix_points."):
                        state_dict["vpmatrix_points"][key.replace("vpmatrix_points.", "")] = f.get_tensor(key)
                    elif key.startswith("atten_modules."):
                        state_dict["atten_modules"][key.replace("atten_modules.", "")] = f.get_tensor(key)
                    elif key.startswith("image_proj_model."):
                        state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.pose_ckpt, map_location="cpu")

        self.image_proj_model.load_state_dict(state_dict["image_proj_model"])
        self.vpmatrix_points.load_state_dict(state_dict["vpmatrix_points"])
        atten_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        atten_layers.load_state_dict(state_dict["atten_modules"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        """ 修改: 这个逻辑应该是通过self.VP矩阵乘self.BasePoints
            输出: image_prompt_embeds, uncond_image_prompt_embeds
            但是之后不需要和原来的text embeds拼接,因为没有text embeds,
            感觉还是有点好,这个后来再看
        """
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    @torch.inference_mode()
    def get_vpmatrix_points(self, V_matrix, P_matrix):
        point_tokens = self.vpmatrix_points(V_matrix, P_matrix)
        uncon_point_tokens = self.vpmatrix_points(torch.zeros_like(V_matrix), torch.zeros_like(P_matrix))
        return point_tokens, uncon_point_tokens


    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, PoseAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        V_matrix=None,
        P_matrix=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)
        
        # 不需要prompt
        # if prompt is None:
        #     prompt = "best quality, high quality"
        # if negative_prompt is None:
        #     negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        # if not isinstance(prompt, List):
        #     prompt = [prompt] * num_prompts
        # if not isinstance(negative_prompt, List):
        #     negative_prompt = [negative_prompt] * num_prompts
        """ 修改:这个 get_image_embeds函数输入不对"""
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        vpmatrix_points_embeds, uncon_vpmatrix_points_embeds= self.get_vpmatrix_points(V_matrix, P_matrix)
        bs_embed, seq_len, _ = vpmatrix_points_embeds.shape
        vpmatrix_points_embeds = vpmatrix_points_embeds.repeat(1, num_samples, 1)
        vpmatrix_points_embeds = vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.repeat(1, num_samples, 1)
        uncon_vpmatrix_points_embeds = uncon_vpmatrix_points_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            """ 修改: 这里到底要不要拼接,原来到底是几维的,中间维度不影响,随便怎么拼"""
            prompt_embeds = torch.cat([vpmatrix_points_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([uncon_vpmatrix_points_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
    

base_point_path=r'/content/drive/MyDrive/PoseCtrl/dataSet/standardVertex.txt'
raw_base_points=load_base_points(base_point_path)  




