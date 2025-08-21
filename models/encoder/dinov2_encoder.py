from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule

from .builder import ENCODERS


@ENCODERS.register_module()
class DINOv2Encoder(BaseModule):
    """DINOv2 Encoder for feature extraction.
    
    Args:
        model_name (str): The DINOv2 model type, e.g., 'dinov2_vitb14'.
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels.
    """
    _model_match = {'small': 'dinov2_vits14', 'basic': 'dinov2_vitb14', 'large': 'dinov2_vitl14'}
    _dinov2_out_channels = {'small': 384, 'basic': 768, 'large': 1024}
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 256,
                 tensor_resize: int = 448,
                 net_type: str = 'small',
                 freeze_dino: bool = True,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', self._model_match[net_type])
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.tensor_resize = tensor_resize
        self.out_layers = build_conv_layer(
                conv_cfg, self._dinov2_out_channels[net_type], out_channels, kernel_size=1)
        if freeze_dino:
            for m in self.dinov2_model.modules():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.in_channels
        x = F.interpolate(x, size=(self.tensor_resize, self.tensor_resize), mode='bilinear', align_corners=False)
        features_dict = self.dinov2_model.forward_features(x)
        patch_tokens = features_dict['x_norm_patchtokens']
        B, N, feat_dim = patch_tokens.shape
        patch_h = patch_w = int(N ** 0.5)
        patch_features = patch_tokens.reshape(B, patch_h, patch_w, feat_dim).permute(0, 3, 1, 2)    # .contiguous()
        out = self.out_layers(patch_features)
        return out, patch_features
        