import torch
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import sys
sys.path.append('/content/drive/MyDrive/PoseCtrl')
sys.path.append('/content/drive/MyDrive/PoseCtrl/poseCtrl')
from poseCtrl.models.pose_adaptor import VPmatrixPoints, ImageProjModel
from poseCtrl.models.attention_processor import AttnProcessor, PoseAttnProcessor
from poseCtrl.data.dataset import CustomDataset, load_base_points
from poseCtrl.models.posectrl import PoseCtrl



base_point_path=r'/content/drive/MyDrive/PoseCtrl/dataSet/standardVertex.txt'
raw_base_points=load_base_points(base_point_path)  

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

# load SD pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

path = r"F:\\Projects\\diffusers\\ProgramData\\sample_new"
dataset = CustomDataset(path)
data = dataset[0]

image = data['image']
image.resize((256, 256))
g_image = data['feature']
g_image.resize((256, 256))

vmatrix = data['view_matrix'].unsqueeze(0)
pmatrix = data['projection_matrix'].unsqueeze(0)

pose_model = PoseCtrl(pipe, image_encoder_path, ip_ckpt, device)
images = pose_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=0.6)
grid = image_grid(images, 1, 4)
grid



