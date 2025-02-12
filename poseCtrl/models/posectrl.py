from PoseCtrl.poseCtrl.models.pose_adaptor import VPmatrixEncoder
import torch
import torch.nn as nn

class posectrl(nn.Module):
    def __init__(self, unet, VPmatrixEncoder, atten_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.VPmatrixEncoder = VPmatrixEncoder
        self.atten_modules = atten_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.VPmatrixEncoder(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_VPmatrix_sum = torch.sum(torch.stack([torch.sum(p) for p in self.VPmatrixEncoder.parameters()]))
        orig_atten_sum = torch.sum(torch.stack([torch.sum(p) for p in self.atten_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.VPmatrixEncoder.load_state_dict(state_dict["VPmatrixEncoder"], strict=True)
        self.atten_modules.load_state_dict(state_dict["atten_modules"], strict=True)

        # Calculate new checksums
        new_VPmatrix_sum = torch.sum(torch.stack([torch.sum(p) for p in self.VPmatrixEncoder.parameters()]))
        new_atten_sum = torch.sum(torch.stack([torch.sum(p) for p in self.atten_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_VPmatrix_sum != new_VPmatrix_sum, "Weights of VPmatrixEncoder did not change!"
        assert orig_atten_sum != new_atten_sum, "Weights of atten_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
