from functools import partial
import torch
import torch.nn.functional as F
from torch import distributed as dist, imag
import numpy as np
from mmcv.ops import Correlation

from models.utils.warp import Warp

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

def random_sample(seq, sample_num):
    '''
    Randomly sample 'sample_num' elements from seq, based on torch.randperm
    Note: Use >1.9 version pytorch, https://github.com/pytorch/pytorch/issues/63726  
    '''
    total_num = seq.size(0)
    if sample_num < 0:
        # if sample num < 0, return an empty tensor
        return seq.new_zeros((0, ))
    if total_num > sample_num:
        random_inds = torch.randperm(total_num, device=seq.device)
        smapled_inds = random_inds[:sample_num]
        return seq[smapled_inds]
    else:
        return seq
    

def reduce_mean(tensor):
    if not (dist.is_initialized() and dist.is_available()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def tensor_image_to_cv2(images:torch.Tensor):
    if images.ndim == 4:
        images = images.permute(0, 2, 3, 1).cpu().data.numpy()
        images = np.ascontiguousarray(images[..., ::-1] * 255).astype(np.uint8)
        return images
    elif images.ndim == 3:
        images = images.cpu().data.numpy()
        images = np.ascontiguousarray(images * 255).astype(np.uint8)
        return images


def simple_forward_warp(images, flow, mask, background_color=(0.5, 0.5, 0.5)):
    warped_images = torch.zeros_like(images)
    warped_images[:, 0] = background_color[0]
    warped_images[:, 1] = background_color[1]
    warped_images[:, 2] = background_color[2]
    height, width = images.size(2), images.size(3)
    num_images = len(images)
    for i in range(num_images): 
        mask_i, flow_i, image_i = mask[i], flow[i], images[i]
        points_y, points_x = torch.nonzero(mask_i, as_tuple=True)
        points_flow = flow_i[:, mask_i.to(torch.bool)]
        warped_x, warped_y = points_x+points_flow[0, :], points_y+points_flow[1, :]
        points_color = image_i[:, mask_i.to(torch.bool)]
        warped_y = torch.clamp(warped_y, min=0, max=height-1)
        warped_x = torch.clamp(warped_x, min=0, max=width-1)
        warped_images[i, :, warped_y.to(torch.int64), warped_x.to(torch.int64)] = points_color
    return warped_images

def rgb_to_gray(tensor):
    """
    Convert an RGB tensor of shape (N, 3, H, W) to a grayscale tensor with 3 identical channels.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, 3, H, W), assumed to be on CUDA.
    
    Returns:
        torch.Tensor: Grayscale tensor of shape (N, 3, H, W) with identical channels.
    """
    # Ensure the input tensor is on CUDA
    assert tensor.is_cuda, "Input tensor must be on CUDA"
    assert tensor.size(1) == 3, "Input tensor must have 3 channels (RGB)"
    
    # Define weights for RGB to grayscale conversion
    weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device, dtype=tensor.dtype)
    
    # Compute grayscale image (N, H, W)
    gray = torch.tensordot(tensor, weights, dims=([1], [0]))
    
    # Expand to 3 channels (N, 3, H, W)
    gray_three_channel = gray.unsqueeze(1).repeat(1, 3, 1, 1)
    
    return gray_three_channel


def dilate_mask_cuda(mask: torch.Tensor, kernel_size: int = 39, iterations: int = 1) -> torch.Tensor:
    assert mask.ndim == 3, "Input tensor must have shape (N, H, W)"
    mask = mask.unsqueeze(1).float()
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    with torch.no_grad():
        for _ in range(iterations):
            dilated = F.conv2d(mask, kernel, padding=kernel_size // 2)
            mask = (dilated > 0).float()
    return mask.squeeze(1)

# import warp as wp
# wp.init()
# @wp.kernel(enable_backward=False)
# def bilateral_filter_depth_kernel(depth:wp.array(dtype=float, ndim=2), out:wp.array(dtype=float, ndim=2), radius:int, zfar:float, sigmaD:float, sigmaR:float):
#     h,w = wp.tid()
#     H = depth.shape[0]
#     W = depth.shape[1]
#     if w>=W or h>=H:
#         return
#     out[h,w] = 0.0
#     mean_depth = float(0.0)
#     num_valid = int(0)
#     for u in range(w-radius, w+radius+1):
#         if u<0 or u>=W:
#             continue
#         for v in range(h-radius, h+radius+1):
#             if v<0 or v>=H:
#                 continue
#             cur_depth = depth[v,u]
#             if cur_depth>=0.1 and cur_depth<zfar:
#                 num_valid += 1
#                 mean_depth += cur_depth
#     if num_valid==0:
#         return
#     mean_depth /= float(num_valid)

#     depthCenter = depth[h,w]
#     sum_weight = float(0.0)
#     sum = float(0.0)
#     for u in range(w-radius, w+radius+1):
#         if u<0 or u>=W:
#             continue
#         for v in range(h-radius, h+radius+1):
#             if v<0 or v>=H:
#                 continue
#         cur_depth = depth[v,u]
#         if cur_depth>=0.1 and cur_depth<zfar and abs(cur_depth-mean_depth)<0.01:
#             weight = wp.exp( -float((u-w)*(u-w) + (h-v)*(h-v)) / (2.0*sigmaD*sigmaD) - (depthCenter-cur_depth)*(depthCenter-cur_depth)/(2.0*sigmaR*sigmaR) )
#             sum_weight += weight
#             sum += weight*cur_depth
#     if sum_weight>0 and num_valid>0:
#         out[h,w] = sum/sum_weight

# @wp.kernel(enable_backward=False)
# def erode_depth_kernel(depth:wp.array(dtype=float, ndim=2), out:wp.array(dtype=float, ndim=2), radius:int, depth_diff_thres:float, ratio_thres:float, zfar:float):
#     h,w = wp.tid()
#     H = depth.shape[0]
#     W = depth.shape[1]
#     if w>=W or h>=H:
#         return
#     d_ori = depth[h,w]
#     if d_ori<0.1 or d_ori>=zfar:
#         out[h,w] = 0.0
#     bad_cnt = float(0)
#     total = float(0)
#     for u in range(w-radius, w+radius+1):
#         if u<0 or u>=W:
#             continue
#         for v in range(h-radius, h+radius+1):
#             if v<0 or v>=H:
#                 continue
#             cur_depth = depth[v,u]
#             total += 1.0
#             if cur_depth<0.1 or cur_depth>=zfar or abs(cur_depth-d_ori)>depth_diff_thres:
#                 bad_cnt += 1.0
#     if bad_cnt/total>ratio_thres:
#         out[h,w] = 0.0
#     else:
#         out[h,w] = d_ori

# def erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100, device='cuda'):
#     depth_wp = wp.from_torch(depth)
#     out_wp = wp.zeros(depth.shape, dtype=float, device=device)
#     wp.launch(kernel=erode_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]], inputs=[depth_wp, out_wp, radius, depth_diff_thres, ratio_thres, zfar],)
#     depth_out = wp.to_torch(out_wp)

#     if isinstance(depth, np.ndarray):
#         depth_out = depth_out.data.cpu().numpy()
#     return depth_out

# def bilateral_filter_depth(depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device='cuda'):
#     if isinstance(depth, np.ndarray):
#         depth_wp = wp.array(depth, dtype=float, device=device)
#     else:
#         depth_wp = wp.from_torch(depth)
#     out_wp = wp.zeros(depth.shape, dtype=float, device=device)
#     wp.launch(kernel=bilateral_filter_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]], inputs=[depth_wp, out_wp, radius, zfar, sigmaD, sigmaR])
#     depth_out = wp.to_torch(out_wp)

#     if isinstance(depth, np.ndarray):
#         depth_out = depth_out.data.cpu().numpy()
#     return depth_out