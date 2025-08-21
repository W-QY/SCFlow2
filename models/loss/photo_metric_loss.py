import torch
from torch import nn
from torch.nn import functional as F
from kornia.filters import get_gaussian_kernel2d, filter2d
from kornia.color import rgb_to_lab
from torch.nn.modules import loss
from typing import Optional
from .builder import LOSSES
from .utils import weighted_loss, tensor_denormalize
from kornia.color import rgb_to_grayscale



def hox_downsample(img):
    '''
    Downsample images with factor equal to 0.5
    Modified from 
    https://github.com/open-mmlab/mmgeneration/blob/master/mmgen/core/evaluation/metric_utils.py
    Args:
        img (tensor): shape (B, C, H, W)
    return:
        img (tensor): downsampled image tensor, shape (B, C, H, W)
    '''
    return (img[:, :, 0::2, 0::2] + img[:, :, 1::2, 0::2] + 
            img[:, :, 0::2, 1::2] + img[:, :, 1::2, 1::2,]) * 0.25




def ssim_metric(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, window_sigma:float = 1.5, max_val: float = 1.0, eps: float = 1e-12
) -> torch.Tensor:
    '''
    Modified from 
    https://github.com/kornia/kornia/blob/master/kornia/metrics/ssim.py
    Args:
        img1 (tensor): the first input image, with shape (B, C, H, W)
        img2 (tensor): the second input image, with shape (B, C, H, W)
        window_size (int): the size of the gaussian kernel to compute the local statistics, default 11
        window_sigma (float): the sigma of the gaussian kernel, default 1.5
        max_val (float): the dynamic range of the images, default 1.0
        eps: Small value for numerically stability when dividing.
    return:
        ssim: (tensor): ssim metric of the two input images, shape (B, C, H, W)
        cs: (tensor): internel C*S for ms-ssim, shape (B, C, H, W)
    '''
    # prepare kernel
    kernel = get_gaussian_kernel2d((window_size, window_size), (window_sigma, window_sigma)).unsqueeze(0)

    # compute coefficients
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # compute local mean per channel
    mu1 = filter2d(img1, kernel)
    mu2 = filter2d(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # compute local sigma per channel
    sigma1_sq = filter2d(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = filter2d(img2 ** 2, kernel) - mu2_sq
    sigma12 = filter2d(img1 * img2, kernel) - mu1_mu2

    # compute the similarity index map
    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    cs = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2 + eps)

    ssim = num / (den + eps)

    return ssim, cs


@weighted_loss
def ssim_loss(pred, target, window_size, window_sigma, max_val, eps):
    ssim, cs = ssim_metric(pred, target, window_size, window_sigma, max_val, eps)
    return 1 - ssim



@weighted_loss
def ms_ssim_loss(pred, target, window_size, window_sigma, level_weights, max_val=1.0, eps=1e-12):
    levels = level_weights.size(0)
    mssim, mcs = [], []
    sigma = window_sigma
    for _ in range(levels):
        ssim, cs = ssim_metric(
            pred, 
            target,
            window_size,
            sigma,
            max_val,
            eps
        )
        mssim.append(ssim.mean(axis=(1, 2, 3)))
        mcs.append(cs.mean(axis=(1, 2, 3)))
        _, _, height, width = pred.shape
        size = min(window_size, height, width)
        sigma = size * window_sigma / window_size
        pred, target = [hox_downsample(img) for img in (pred, target)]
    
    # shape (num_level, B)
    mssim = torch.clamp(torch.stack(mssim, axis=0), 0.0)
    # shape (num_level, B)
    mcs = torch.clamp(torch.stack(mcs, axis=0), 0.0)

    ms_ssim = torch.prod(mcs[:-1, :] ** level_weights[:-1, None], dim=0) * mssim[-1, :] ** level_weights[-1]
    return 1 - ms_ssim

    





@LOSSES.register_module()
class SSIMLoss(nn.Module):
    def __init__(self,
                window_size=11,
                window_sigma=1.5,
                max_val=1.0,
                eps=1e-12,
                reduction='mean',
                loss_weight=1.0):
        super().__init__()
        self.window_size = window_size
        self.window_sigma = window_sigma
        self.max_val = max_val
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, 
                image_pred,
                image_gt,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override is not None else self.reduction)
        loss = self.loss_weight * ssim_loss(
            image_pred, 
            image_gt, 
            weight, 
            reduction=reduction, 
            avg_factor=avg_factor, 
            window_size=self.window_size,
            window_sigma=self.window_sigma,
            max_val=self.max_val,
            eps=self.eps)
        return loss

@LOSSES.register_module()
class MSSSIMLoss(nn.Module):
    def __init__(self,
                window_size=11,
                window_sigma=1.5,
                max_val=1.0,
                eps=1e-12,
                level_weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                reduction='mean',
                loss_weight=1.0):
        super().__init__()
        self.max_val = max_val
        self.window_size = window_size
        self.window_sigma = window_sigma
        self.eps = eps
        self.level_weights = nn.Parameter(torch.tensor(level_weights), requires_grad=False)
        self.redcution = reduction
        self.loss_weight = loss_weight

    def forward(self, 
                image_pred,
                image_gt,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override is not None else self.redcution
        )

        loss = self.loss_weight * ms_ssim_loss(
            image_pred,
            image_gt,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            window_size=self.window_size,
            window_sigma=self.window_sigma,
            level_weights=self.level_weights,
            max_val=self.max_val,
            eps=self.eps
        )
        return loss

    


@weighted_loss
def psnr_loss(pred, target, max_val):
    mse = F.mse_loss(pred, target, reduction='none')
    mse = torch.mean(mse, dim=(1, 2, 3))
    return 100 - 10.0 * torch.log10(max_val ** 2 / mse)

@LOSSES.register_module()
class PSNRLoss(nn.Module):
    def __init__(self,
                max_val=1.0,
                reduction='mean',
                loss_weight=1.0):
        super().__init__()
        self.max_val = max_val
        self.reduction = reduction
        self.loss_weight = loss_weight

    
    def forward(self, 
                image_pred,
                image_gt,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override is not None else self.reduction
        )

        return self.loss_weight * psnr_loss(
            image_pred, 
            image_gt,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            max_val=self.max_val
        )




@weighted_loss
def mse_loss(pred, target):
    """Warpper of mse loss.
    
    """
    mse = F.mse_loss(pred, target, reduction='none')
    return mse

@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss




def census_transform(image, window_size):
    '''
    Modified from https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py
    '''
    kernel = torch.eye(window_size * window_size, device=image.device)
    kernel = torch.reshape(kernel, (window_size*window_size, 1, window_size, window_size))
    neighbors = F.conv2d(image, kernel, stride=1, padding='same')
    diff = neighbors - image
    diff_norm = diff / torch.sqrt(.81 + torch.square(diff))
    return diff_norm

@weighted_loss
def census_loss(image_pred, image_target, window_size, thresh, eps=0.01, q=0.4):
    gray_pred = rgb_to_grayscale(image_pred) * 255.
    gray_target = rgb_to_grayscale(image_target) * 255.
    transformed_pred = census_transform(gray_pred, window_size)
    transformed_target = census_transform(gray_target, window_size)
    # soft hamming distance
    sq_dist = torch.square(transformed_pred - transformed_target)
    soft_thresh_dist = sq_dist / (thresh + sq_dist)
    soft_thresh_dist = torch.sum(soft_thresh_dist, dim=1, keepdim=True)
    # charbonnier loss
    loss = torch.pow(torch.abs(soft_thresh_dist) + eps, q)
    return loss




@LOSSES.register_module()
class CensusLoss(nn.Module):
    def __init__(self,
                window_size=7,
                thresh=.1, 
                eps=0.01,
                q=0.4,
                reduction='mean',
                loss_weight=1.0):
        super().__init__()
        self.window_size = window_size
        self.thresh = thresh
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.q = q

    
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        '''
        Forward function of loss
        Args;
            pred (torch.tensor): The predicted image, shape (B, C, H, W)
            target (torch.tensor): The target image, shape (B, C, H, W)
            weight (torch.tensor, optional): Weight of the loss for each location,
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to avaerage the loss,
                Defaults to None.
             
        '''
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = self.loss_weight * census_loss(
            pred,
            target,
            weight,
            window_size=self.window_size,
            thresh=self.thresh,
            eps=self.eps,
            q=self.q,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss 

@LOSSES.register_module()
class LABSpaceLoss(nn.Module):
    def __init__(self, loss_weight=1.0, with_L:bool=False) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.with_L = with_L
    

    def forward(
        self, tgt_images:torch.Tensor, src_images:torch.Tensor, 
        valid:Optional[torch.Tensor]=None, img_norm_cfg:Optional[dict]=None):
        if img_norm_cfg is not None:
            tgt_images = tensor_denormalize(tgt_images, img_norm_cfg['mean'], img_norm_cfg['std'])/255
            src_images = tensor_denormalize(src_images, img_norm_cfg['mean'], img_norm_cfg['std'])/255
        tgt_lab_images, src_lab_images = rgb_to_lab(tgt_images), rgb_to_lab(src_images)
        if self.with_L:
            loss = torch.abs(tgt_lab_images - src_lab_images).mean(dim=1)
        else:
            loss = torch.abs(tgt_lab_images[:, 1:, ...] - src_lab_images[:, 1:, ...]).mean(dim=1)
        
        if valid is None:
            valid = torch.ones_like(loss)
        
        loss = torch.sum(loss * valid) / max(1, valid.sum()) * self.loss_weight
        return loss