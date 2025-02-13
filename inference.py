import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from poseCtrl.data.dataset import MyDataset
from poseCtrl.models.posectrl import posectrl
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter, DDIMScheduler, AutoencoderKL


""" 修改 """

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)


# load t2i-adapter
adapter_model_path = "diffusers/t2iadapter_depth_sd15v2/"
adapter = T2IAdapter.from_pretrained(adapter_model_path, torch_dtype=torch.float16)
# load SD pipeline
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    base_model_path,
    adapter=adapter,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# read image prompt
image = Image.open("assets/images/river.png")
depth_map = Image.open("assets/structure_controls/depth2.png")
image_grid([image.resize((256, 256)), depth_map.resize((256, 256))], 1, 2)


# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42)
grid = image_grid(images, 1, 4)


