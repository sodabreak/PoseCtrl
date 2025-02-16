import torch
from pathlib import Path



def change_checkpoint(checkpoint_path, new_checkpoint_path):
    sd = torch.load(checkpoint_path, map_location="cpu")
    vpmatrix_points_sd = {}
    atten_sd = {}
    proj_sd={}
    for k in sd:
        if k.startswith("unet"):
            pass
        elif k.startswith("vpmatrix_points"):
            vpmatrix_points_sd[k.replace("vpmatrix_points.", "")] = sd[k]
        elif k.startswith("atten_modules"):
            atten_sd[k.replace("atten_modules.", "")] = sd[k]
        elif k.startswith("image_proj_model"):
            proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    new_checkpoint_path = Path(new_checkpoint_path, "posectrl.bin")
    torch.save({"vpmatrix_points": vpmatrix_points_sd, "atten_modules": atten_sd, "image_proj_model": proj_sd}, new_checkpoint_path)
    print(f"Saved new checkpoint to {new_checkpoint_path}")



def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator