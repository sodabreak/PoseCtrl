import torch
from pathlib import Path



def change_checkpoint(checkpoint_path, new_checkpoint_path):
    sd = torch.load(checkpoint_path, map_location="cpu")
    VPmatrixEncoder_sd = {}
    atten_sd = {}
    for k in sd:
        if k.startswith("unet"):
            pass
        elif k.startswith("VPmatrixEncoder"):
            VPmatrixEncoder_sd[k.replace("VPmatrixEncoder.", "")] = sd[k]
        elif k.startswith("atten_modules"):
            atten_sd[k.replace("atten_modules.", "")] = sd[k]
    new_checkpoint_path = Path(new_checkpoint_path, "posectrl.bin")
    torch.save({"VPmatrixEncoder": VPmatrixEncoder_sd, "atten_modules": atten_sd}, new_checkpoint_path)
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