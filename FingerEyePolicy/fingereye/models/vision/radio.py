import torch
import torch.nn as nn
from einops import rearrange
from torchvision import transforms


class RADIO(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()
        self.radio = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version="radio_v2.5-b",
            progress=True,
            skip_validation=True,
        )
        self.radio.eval()
        for param in self.radio.parameters():
            param.requires_grad = False
        self.patch_size = 16
        self.dim = 768
        self.summary_dim = 2304

    def image_center_crop(self, image_tensor):
        """
        Args:
            image_tensor: shape [B, 3, H, W]
        Returns:
            image_tensor: shape [B, 3, H', W']
        """
        B, _, H, W = image_tensor.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            patch_h = int(H // self.patch_size)
            patch_w = int(W // self.patch_size)
            new_H = patch_h * self.patch_size
            new_W = patch_w * self.patch_size
            image_tensor = transforms.CenterCrop((new_H, new_W))(image_tensor)
        return image_tensor

    # @torch.no_grad()
    def get_feature_grid(self, image_tensor, return_processed_img=True):
        """
        Args:
            image_tensor: shape [B, 3, H, W] | range [0, 1]
        Returns:
            feature_grid: shape [B, patch_h, patch_w, feature_dim]
        """
        image_tensor = self.image_center_crop(image_tensor)

        summary, spatial_features = self.radio(image_tensor, feature_fmt="NCHW")
        feature_grid = rearrange(spatial_features, "b c h w -> b h w c") # b 320/16 240/16  self.dim

        if return_processed_img:
            return feature_grid, image_tensor
        else:
            return feature_grid, summary
