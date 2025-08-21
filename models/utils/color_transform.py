from sys import flags
from typing import Optional, Dict
from mmcv.utils import Registry, build_from_cfg
import kornia
import torch
from torch import Tensor
from kornia.augmentation.base import IntensityAugmentationBase2D
from kornia.filters import box_blur
from kornia.color import rgb_to_hsv
import random

Augmemtations = Registry('augmentations')


def build_augmentation(cfg):
    return build_from_cfg(cfg, Augmemtations)


@Augmemtations.register_module()
class RandomNoise(IntensityAugmentationBase2D):
    def __init__(self, 
                noise_ratio:0.1,
                same_on_batch: bool = False,
                p: float = 0.5,
                keepdim: bool = False,
                return_transform: Optional[bool] = None) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )
        self.noise_ratio = noise_ratio
    
    def generate_parameters(self, shape: torch.Size) -> Dict[str, Tensor]:
        noise_sigma = random.uniform(0, self.noise_ratio)
        noise = torch.normal(mean=0., std=noise_sigma, size=shape)
        return dict(noise=noise)
    
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ) -> Tensor:
        return torch.clamp_(input + params["noise"].to(input.device), min=0., max=1.)

@Augmemtations.register_module()
class RandomSmooth(IntensityAugmentationBase2D):
    def __init__(self, 
                max_kernel_size=5,
                border_type: str = "reflect",
                normalized: bool = True,
                return_transform: bool = None, 
                same_on_batch: bool = False, 
                p: float = 0.5, 
                p_batch: float = 1, 
                keepdim: bool = False) -> None:
        super().__init__(return_transform, same_on_batch, p, p_batch, keepdim)
        max_kernel_size = int(max_kernel_size)
        kernel_sizes = [i*2+1 for i in range(max_kernel_size//2+1)]
        self.flags = dict(kernel_sizes=kernel_sizes, border_type=border_type, normalized=normalized)

    
    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor]) -> Tensor:
        return self.identity_matrix(input)
    
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ) -> Tensor:
        kernel_size = random.choice(self.flags['kernel_sizes'])
        return box_blur(input, (kernel_size, kernel_size), self.flags["border_type"], self.flags["normalized"])

@Augmemtations.register_module()
class RandomHSV(IntensityAugmentationBase2D):
    def __init__(self, 
                h_ratio, 
                s_ratio, 
                v_ratio,
                return_transform: bool = None, 
                same_on_batch: bool = False, 
                p: float = 0.5, 
                p_batch: float = 1, 
                keepdim: bool = False) -> None:
        super().__init__(return_transform, same_on_batch, p, p_batch, keepdim)
        self.h_ratio = h_ratio
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio
    
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hsv = rgb_to_hsv(input)
        h, s, v = torch.chunk(input, chunks=3, dim=-3)
        h = h * (random.uniform(-1, 1) * self.h_ratio + 1)
        s = s * (random.uniform(-1, 1) * self.s_ratio + 1)
        v = v * (random.uniform(-1, 1) * self.v_ratio + 1)
        return torch.cat([h, s, v], dim=-3)
