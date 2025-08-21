import torch
from torch.nn import functional as F
from .warp import coords_grid, Warp


def filter_flow_by_mask(flow, gt_mask, invalid_num=400, mode='bilinear', align_corners=False):
    '''Check if flow is valid. 
    When the flow pointed point not in the target image mask or falls out of the target image, the flow is invalid.
    Args:
        flow (tensor): flow from source image to target image, shape (N, 2, H, W)
        mask (tensor): mask of the target image, shape (N, H, W)
    '''
    not_valid_mask = (flow[:, 0] >= invalid_num) & (flow[:, 1] >= invalid_num)
    mask = gt_mask[:, None].to(flow.dtype)
    grid = coords_grid(flow)
    mask = F.grid_sample(
        mask,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=align_corners
    )
    not_valid_mask = (mask < 0.9) | not_valid_mask[:, None]
    not_valid_mask = not_valid_mask.expand_as(flow)
    flow[not_valid_mask] = invalid_num
    return flow

def filter_flow_by_depth(
    flow:torch.Tensor, depth1:torch.Tensor, depth0:torch.Tensor, invalid_num=400, thr=0.2):
    # flow is from image 0 to image 1
    # https://github.com/zju3dv/LoFTR/blob/master/src/loftr/utils/geometry.py
    not_valid_mask = (flow[:, 0] >= invalid_num) & (flow[:, 1] >= invalid_num)
    mask0, mask1 = depth0 > 0, depth1 > 0
    grid = coords_grid(flow)
    depth1_masked, depth0_masked = depth1.clone(), depth0.clone()
    depth1_masked[~mask1.bool()] = 0.
    depth0_masked[~mask0.bool()] = 0.
    depth1_expanded, depth0_expanded = depth1_masked[:, None], depth0_masked[:, None]
    warped_depth = F.grid_sample(depth1_expanded, grid, padding_mode='zeros', mode='bilinear', align_corners=True)
    consistent_mask = ((depth0_expanded - warped_depth).abs() / (depth0_expanded + 0.1)) < thr
    
    not_valid_mask = not_valid_mask[:, None] & (~ consistent_mask)
    not_valid_mask = not_valid_mask.expand_as(flow)
    flow[not_valid_mask] = invalid_num
    return flow

def filter_flow_by_face_index(
    flow:torch.Tensor, face_index1:torch.Tensor, face_index2:torch.Tensor, invalid_num=400):
    not_valid_mask = (flow[:, 0] >= invalid_num) & (flow[:, 1] >= invalid_num)
    face_index1, face_index2 = face_index1.to(torch.float32), face_index2.to(torch.float32)
    grid = coords_grid(flow)
    face_index1_expanded, face_index2_expanded = face_index1[:, None], face_index2[:, None]
    warped_face_index2 = F.grid_sample(face_index2_expanded, grid, padding_mode='zeros', mode='nearest', align_corners=True)
    consisent_mask = warped_face_index2 == face_index1_expanded
    
    not_valid_mask = not_valid_mask[:, None] | (~ consisent_mask)
    not_valid_mask = not_valid_mask.expand_as(flow)
    flow[not_valid_mask] = invalid_num
    return flow

def multiview_variance_mean(
    base_view_flow:torch.Tensor, other_view_flow:torch.Tensor, 
    view_flow:torch.Tensor, base_view_flow_valid_mask:torch.Tensor, 
    invalid_num:int=400, valid_view_num_thr:int=2, gaussian_fitting:bool=False):
    EPS, INF = 1e-10, 1e10
    warp_op = Warp(use_mask=True, return_mask=False)
    other_view_coords = flow_to_coords(other_view_flow)
    base_view_coords = flow_to_coords(base_view_flow[None])
    warped_other_view_coords = warp_op(other_view_coords, view_flow) # (N-1, 2, H, W)
    valid_mask_per_view = (view_flow[:, 0] < invalid_num) | (view_flow[:, 1] < invalid_num) # (N-1, H, W)
    view_coords = torch.cat([base_view_coords, warped_other_view_coords], dim=0) # (N, 2, H, W)
    valid_mask_per_view = torch.cat([base_view_flow_valid_mask[None], valid_mask_per_view], dim=0) # (N, H, W)
    valid_mask = torch.sum(valid_mask_per_view, dim=0) > valid_view_num_thr
    valid_view_num_per_pixel = torch.sum(valid_mask_per_view, dim=0)
    view_coords_mean = torch.sum(view_coords*valid_mask_per_view[:, None], dim=0, keepdim=False) \
                    / (valid_view_num_per_pixel + EPS)
    
    valid_view_flag = valid_mask_per_view[:, valid_mask].to(torch.bool)
    valid_points_view_coords = view_coords[:, :, valid_mask]
    valid_points_coords_mean = view_coords_mean[:, valid_mask]
    valid_points_coords_var = torch.sum((valid_points_view_coords - valid_points_coords_mean[None])**2, dim=1)
    valid_points_coords_var[~valid_view_flag] = 0
    valid_points_coords_var = torch.sqrt(torch.sum(valid_points_coords_var, dim=0) / torch.sum(valid_view_flag, dim=0))
    
    view_coords_var = torch.full_like(valid_mask, fill_value=INF, dtype=torch.float32)
    view_coords_var[valid_mask] = valid_points_coords_var

    view_flow_mean = view_coords_mean - flow_to_coords(torch.zeros_like(base_view_flow[None]))[0]

    if gaussian_fitting:
        var_std = torch.std(torch.cat([-valid_points_coords_var, valid_points_coords_var]))
        gaussian_dis = norm(loc=0, scale=var_std.cpu().numpy())
        valid_gaussian_weights = gaussian_dis.pdf(valid_points_coords_var.cpu().numpy())
        gaussian_weights = torch.zeros_like(valid_mask, dtype=torch.float32)
        gaussian_weights[valid_mask] = valid_gaussian_weights
        return view_flow_mean, view_coords_var, gaussian_weights
    else:
        return view_flow_mean, view_coords_var




def cal_epe(flow_tgt, flow_pred, mask, max_flow=400, reduction='mean', threshs=[1, 3, 5]):
    mag = torch.sum(flow_tgt**2, dim=1).sqrt()
    if mask is not None:
        # filter the noisy sample with too large flow and without explicit correspondence
        valid_mask = ((mag < max_flow) & (mask >=0.5))
    else:
        valid_mask = (mag< max_flow)
    flow_error = torch.sum((flow_tgt - flow_pred)**2, dim=1).sqrt()
    if reduction == 'none':
        flow_error = flow_error * valid_mask.to(flow_error)
        return flow_error
    elif reduction == 'mean':
        flow_acc = dict()
        total_valid_pixel_num = valid_mask.sum(dim=(-1, -2)) + 1e-10
        flow_acc['mean'] = (flow_error * valid_mask.to(flow_error)).sum(dim=(-1, -2)) / total_valid_pixel_num
        flow_error[valid_mask] = 1e+8
        for thresh in threshs:
            flow_acc[f'{thresh}px'] = (flow_error < thresh).sum(dim=(-1, -2))/ total_valid_pixel_num 
    elif reduction  == 'total_mean':
        flow_acc = dict()
        total_valid_pixel_num = valid_mask.sum(dim=(-1, -2, -3)) + 1e-10
        flow_acc['mean'] = (flow_error * valid_mask.to(flow_error.dtype)).sum(dim=(-1,-2,-3)) / total_valid_pixel_num
        for thresh in threshs:
            flow_acc[f'{thresh}px'] = (flow_error[valid_mask] < thresh).sum() / total_valid_pixel_num
    return flow_acc

def flow_to_coords(flow: torch.Tensor):
    """Generate shifted coordinate grid based based input flow.
    Args:
        flow (Tensor): Estimated optical flow.
    Returns:
        Tensor: Coordinate that shifted by input flow with shape (B, 2, H, W).
    """
    B, _, H, W = flow.shape
    xx = torch.arange(0, W, device=flow.device, requires_grad=False)
    yy = torch.arange(0, H, device=flow.device, requires_grad=False)
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(B, 1, 1, 1) + flow
    return coords
    


def compute_range_map(flow: torch.Tensor) -> torch.Tensor:
    """Compute range map.
    Args:
        flow (Tensor): The backward flow with shape (N, 2, H, W)
    Return:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """

    N, _, H, W = flow.shape

    coords = flow_to_coords(flow)

    # Split coordinates into an integer part and
    # a float offset for interpolation.
    coords_floor = torch.floor(coords)
    coords_offset = coords - coords_floor
    coords_floor = coords_floor.to(torch.int32)

    # Define a batch offset for flattened indexes into all pixels.
    batch_range = torch.arange(N).view(N, 1, 1)
    idx_batch_offset = batch_range.repeat(1, H, W) * H * W

    # Flatten everything.
    coords_floor_flattened = coords_floor.permute(0, 2, 3, 1).reshape(-1, 2)
    coords_offset_flattened = coords_offset.permute(0, 2, 3, 1).reshape(-1, 2)
    idx_batch_offset_flattened = idx_batch_offset.reshape(-1)

    # Initialize results.
    idxs_list = []
    weights_list = []

    # Loop over differences di and dj to the four neighboring pixels.
    for di in range(2):
        for dj in range(2):
            # Compute the neighboring pixel coordinates.
            idxs_j = coords_floor_flattened[..., 0] + dj
            idxs_i = coords_floor_flattened[..., 1] + di
            # Compute the flat index into all pixels.
            idxs = idx_batch_offset_flattened + idxs_i * W + idxs_j

            # Only count valid pixels.
            mask = torch.logical_and(
                torch.logical_and(idxs_j >= 0, idxs_j < W),
                torch.logical_and(idxs_i >= 0, idxs_i < H))
            valid_idxs = idxs[mask]
            valid_offsets = coords_offset_flattened[mask]

            # Compute weights according to bilinear interpolation.
            weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 0]
            weights_i = (1. - di) - (-1)**di * valid_offsets[:, 1]
            weights = weights_i * weights_j

            # Append indices and weights to the corresponding list.
            idxs_list.append(valid_idxs)
            weights_list.append(weights)
    # Concatenate everything.
    idxs = torch.cat(idxs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)

    # Sum up weights for each pixel and reshape the result.
    count_image = torch.zeros(N * H * W)
    count_image = count_image.index_add_(
        dim=0, index=idxs, source=weights).reshape(N, H, W)
    occ = (count_image >= 1).to(flow)[:, None, ...]
    return occ


def forward_backward_consistency(
        flow_fw: torch.Tensor,
        flow_bw: torch.Tensor,
        warp_cfg: dict = dict(align_corners=True),
) -> torch.Tensor:
    """Occlusion mask from forward-backward consistency.
    Args:
        flow_fw (Tensor): The forward flow with shape (N, 2, H, W)
        flow_bw (Tensor): The backward flow with shape (N, 2, H, W)
    Returns:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """

    warp = Warp(**warp_cfg)

    warped_flow_bw = warp(flow_bw, flow_fw)

    forward_backward_sq_diff = torch.sum(
        (flow_fw + warped_flow_bw)**2, dim=1, keepdim=True)
    forward_backward_sum_sq = torch.sum(
        flow_fw**2 + warped_flow_bw**2, dim=1, keepdim=True)

    occ = (forward_backward_sq_diff <
           forward_backward_sum_sq * 0.01 + 0.5).to(flow_fw)
    return occ


def forward_backward_absdiff(flow_fw: torch.Tensor,
                             flow_bw: torch.Tensor,
                             warp_cfg: dict = dict(align_corners=True),
                             diff: int = 1.5) -> torch.Tensor:
    """Occlusion mask from forward-backward consistency.
    Args:
        flow_fw (Tensor): The forward flow with shape (N, 2, H, W)
        flow_bw (Tensor): The backward flow with shape (N, 2, H, W)
    Returns:
        Tensor: The forward-to-backward occlusion mask with shape (N, 1, H, W)
    """

    warp = Warp(**warp_cfg)

    warped_flow_bw = warp(flow_bw, flow_fw)

    forward_backward_sq_diff = torch.sum(
        (flow_fw + warped_flow_bw)**2, dim=1, keepdim=True)

    occ = (forward_backward_sq_diff**0.5 < diff).to(flow_fw)

    return occ


def occlusion_estimation(flow_fw: torch.Tensor,
                         flow_bw: torch.Tensor,
                         mode: str = 'consistency',
                         **kwarg):
    """Occlusion estimation.
    Args:
        flow_fw (Tensor): The forward flow with shape (N, 2, H, W)
        flow_bw (Tensor): The backward flow with shape (N, 2, H, W)
        mode (str): The method for occlusion estimation, which can be
            ``'consistency'``, ``'range_map'`` or ``'fb_abs'``.
        warp_cfg (dict, optional): _description_. Defaults to None.
    Returns:
        Dict[str,Tensor]: 1 denote non-occluded and 0 denote occluded
    """
    assert mode in ('consistency', 'range_map', 'fb_abs'), \
        'mode must be \'consistency\', \'range_map\' or \'fb_abs\', ' \
        f'but got {mode}'

    if mode == 'consistency':
        occ_fw = forward_backward_consistency(flow_fw, flow_bw, **kwarg)
        occ_bw = forward_backward_consistency(flow_bw, flow_fw, **kwarg)

    elif mode == 'range_map':
        occ_fw = compute_range_map(flow_bw)
        occ_bw = compute_range_map(flow_fw)

    elif mode == 'fb_abs':
        occ_fw = forward_backward_absdiff(flow_fw, flow_bw, **kwarg)
        occ_bw = forward_backward_absdiff(flow_bw, flow_fw, **kwarg)

    return dict(occ_fw=occ_fw, occ_bw=occ_bw)