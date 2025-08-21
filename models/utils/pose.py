from typing import Sequence, Optional
import torch
from torch.nn import functional as F
import cv2
import numpy as np
import kornia
from .warp import Warp
from .flow import filter_flow_by_mask, filter_flow_by_depth, filter_flow_by_face_index 
from datasets.pose import remap_pose
from .ransac_kabsch import BatchRansacEstimator

def interpolate_depth(points_x, points_y, depth):
    h, w = depth.shape
    normalized_points_x = points_x * 2 / w - 1.
    normalized_points_y = points_y * 2 / h - 1.
    grid = torch.stack([normalized_points_x, normalized_points_y], dim=-1)
    depth_interploated = F.grid_sample(
        depth[None, None],
        grid[None, None],
        mode='bilinear',
        padding_mode='zeros'
    )
    depth_interploated = depth_interploated[0, 0, 0]
    return depth_interploated
    


def lift_2d_to_3d(
    points_x:torch.Tensor, points_y:torch.Tensor, points_depth:torch.Tensor, 
    internel_k:torch.Tensor, rotation:Optional[torch.Tensor]=None, translation:Optional[torch.Tensor]=None):
    '''Unproject 2d points to 3d
    With only internel_k, return the 3d points defined in the camera frame
    If rotation and translation are given, the 3d points defined in the object frame will also be returned.
    '''
    assert len(points_x) == len(points_y) == len(points_depth)
    homo_points = torch.stack([points_x, points_y, torch.ones_like(points_x)], dim=-1).float()
    points_camera_frame = homo_points * points_depth[..., None]
    points_camera_frame = torch.mm(torch.inverse(internel_k), points_camera_frame.t()).t()
    if rotation is not None and translation is not None:
        points_object_frame = torch.mm(torch.inverse(rotation), (points_camera_frame - translation[None]).t()).t()
        
        # test solve_pose_by_ransac_kabsch
        if False:    
            rotation_pred, translation_pred, flag = solve_pose_by_ransac_kabsch(points_camera_frame, points_object_frame)

        return points_camera_frame, points_object_frame
    else:
        return points_camera_frame


def cal_3d_2d_corr(depth, internel_k, rotation, translation, occlusion=None):
    '''Calculate 2D-3D correspondance
    Args:
        depth (Tensor): shape (H, W)
        internel_k (Tensor): shape (3, 3)
        rotations (Tensor): shape (3, 3)
        translations (Tensor): shape (3)
        occlusioh (Tensor): shape (H, W)
    return:
        points_2d (Tensor): shape (N, 2), foreground 2d points, xy format
        points_3d (Tensor): shape (N, 3), corresponding 3d location, xyz format     
        
    '''
    mask = depth > 0
    if occlusion is not None:
        mask = mask & occlusion
    points2d_y, points2d_x = torch.nonzero(mask, as_tuple=True)
    points_depth = depth[mask]
    _, points3d_object_frame = lift_2d_to_3d(
        points2d_x.float(), points2d_y.float(), points_depth, internel_k, rotation, translation)
    return torch.stack([points2d_x, points2d_y], dim=-1).float(), points3d_object_frame

def get_flow_from_delta_pose_and_points(rotation_dst, translation_dst, k, points_2d_list, points_3d_list, height, width, invalid_num=400.):
    '''Calculate flow from source image to target image
    Args:
        rotation_dst (Tensor): rotation matrix for target image, shape (n, 3, 3)
        translation_dst (Tensor): translation vector for target image, shape (n, 3)
        k (Tensor): camera intrinsic for source image and taregt image, shape (n, 3, 3)
        points_2d_list (Tensor): source image 2d points, (x,y), each element has shape (N, 2), where N is the number of points
        points_3d_list (Tensor): source image 3d points, (x,y,z), each element has shape (N, 3), where N is the number of points
        height, width (int): patch image resolution 
        invalid_num (float): set invalid flow to this number
    '''
    num_images = len(rotation_dst)
    flow = rotation_dst.new_ones((num_images, 2, height, width)) * invalid_num
    for i in range(num_images):
        points_2d, points_3d = points_2d_list[i], points_3d_list[i]
        points_3d_transpose = points_3d.t()
        points_2d_dst = torch.mm(k[i], torch.mm(rotation_dst[i], points_3d_transpose)+translation_dst[i][:, None]).t()
        points_2d_dst_x, points_2d_dst_y = points_2d_dst[:, 0]/points_2d_dst[:, 2], points_2d_dst[:, 1]/points_2d_dst[:, 2]
        flow_x, flow_y = points_2d_dst_x - points_2d[:, 0], points_2d_dst_y - points_2d[:, 1]
        flow = flow.to(flow_x.dtype)
        flow[i, 0, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)] = flow_x
        flow[i, 1, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)] = flow_y
    return flow



def get_flow_from_delta_pose_and_depth(
    rotation_src, translation_src, rotation_dst, translation_dst, depth_src, k, invalid_num=400):
    '''Calculate flow from source image to target image
    Args:
        rotation_src (Tensor): rotation matrix for source image, shape (n, 3, 3)
        translation_src (Tensor): translatio vector for source image, shape (n, 3)
        rotation_dst (Tenosr): rotation matrix for target image, shape (n, 3, 3)
        translation_dst (Tensor): translation vector for target image, shape (n, 3)
        depth_src (Tensor): depth for source image, shape (n, H, W)
        k (Tensor): camera intrinsic for source image and taregt image. 
    return:
        flow (Tensor): flow from the source image to the target image, shape (n, 2, H, W)
    
    '''
    num_images = rotation_src.shape[0]
    height, width = depth_src.shape[-2:]
    flow = depth_src.new_ones((num_images, 2, height, width), ) * invalid_num
    for i in range(num_images):
        points_2d_src, points_3d_src = cal_3d_2d_corr(depth_src[i], k[i], rotation_src[i], translation_src[i])
        points_3d_src_transpose = points_3d_src.t()
        points_2d_dst = torch.mm(k[i], torch.mm(rotation_dst[i], points_3d_src_transpose)+translation_dst[i][:, None]).t()
        points_2d_dst_x, points_2d_dst_y = points_2d_dst[:, 0], points_2d_dst[:, 1]
        points_2d_dst_x = points_2d_dst_x / points_2d_dst[:, 2]
        points_2d_dst_y = points_2d_dst_y / points_2d_dst[:, 2]
        flow_x = points_2d_dst_x - points_2d_src[:, 0]
        flow_y = points_2d_dst_y - points_2d_src[:, 1]
        flow = flow.to(flow_x.dtype)
        flow[i, 0, points_2d_src[:, 1].to(torch.int64), points_2d_src[:, 0].to(torch.int64)] = flow_x
        flow[i, 1, points_2d_src[:, 1].to(torch.int64), points_2d_src[:, 0].to(torch.int64)] = flow_y
    return flow


def get_pose_from_delta_pose(
        rotation_delta, translation_delta, rotation_src, translation_src, 
        focal_x=10., focal_y=10., depth_transform='exp', detach_depth_for_xy=False):
    '''Get transformed pose
    Args:
        rotation_delta (Tensor): quaternion to represent delta rotation shape (n, 4)(Quaternions) or (n, 6)(orth 6D )
        translation_delta (Tensor): translation to represent delta translation shape (n, 3)
        rotation_src (Tensor): rotation matrix to represent source rotation shape (n, 3, 3)
        translation_src (Tensor): translation vector to represent source translation shape (n, 3)
    '''
    if rotation_delta.size(1) == 4:
        rotation_delta = kornia.geometry.conversions.quaternion_to_rotation_matrix(rotation_delta)
    elif rotation_delta.size(1) == 6:
        rotation_delta = get_rotation_matrix_from_ortho6d(rotation_delta)
    else:
        None
    rotation_dst = torch.bmm(rotation_delta, rotation_src)
    if depth_transform == 'exp':
        vz = torch.div(translation_src[:, 2], torch.exp(translation_delta[:, 2]))
    else:
        # vz = torch.div(translation_src[:, 2], translation_delta[:, 2] + 1)
        vz = translation_src[:, 2] * (translation_delta[:, 2] + 1)
    if detach_depth_for_xy:
        vx = torch.mul(vz.detach(), torch.addcdiv(translation_delta[:, 0] / focal_x, translation_src[:, 0], translation_src[:, 2]))
        vy = torch.mul(vz.detach(), torch.addcdiv(translation_delta[:, 1] / focal_y, translation_src[:, 1], translation_src[:, 2]))
    else:
        vx = torch.mul(vz, torch.addcdiv(translation_delta[:, 0] / focal_x, translation_src[:, 0], translation_src[:, 2]))
        vy = torch.mul(vz, torch.addcdiv(translation_delta[:, 1] / focal_y, translation_src[:, 1], translation_src[:, 2]))
    translation_dst = torch.stack([vx, vy, vz], dim=-1)
    return rotation_dst, translation_dst



def get_rotation_matrix_from_ortho6d(ortho6d):
    '''
    https://github.com/papagina/RotationContinuity/blob/sanity_test/code/tools.py L47
    '''
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = F.normalize(x_raw, p=2, dim=1) #batch*3
    z = torch.cross(x, y_raw, dim=1)
    z = F.normalize(z, p=2, dim=1)
    y = torch.cross(z, x, dim=1)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def save_xyzrgb(points_3d, save_path:str, points_2d=None, image=None):
    if save_path.endswith('xyzrgb'):
        assert image is not None
        assert points_2d is not None
        color = image[:, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)]
        save_content = torch.cat([points_3d, color.transpose(0, 1)], dim=1)
    else:
        save_content = points_3d
    np.savetxt(save_path, save_content.cpu().data.numpy())


def get_2d_3d_corr_by_fw_flow(fw_flow, rendered_depths, ref_rotations, ref_translations, internel_k, valid_mask=None):
    '''
    Return a list of tuple, each element has three components, 
        2d points in the ref image, 2d points in the tgt image, and corresponding 3d points 
    '''
    num_images = len(fw_flow)
    points_corr = []
    for i in range(num_images):
        if valid_mask is not None:
            points_2d, points_3d = cal_3d_2d_corr(
                rendered_depths[i], internel_k[i], ref_rotations[i], ref_translations[i], valid_mask[i])
        else:
            points_2d, points_3d = cal_3d_2d_corr(
                    rendered_depths[i], internel_k[i], ref_rotations[i], ref_translations[i])
        pred_flow = fw_flow[i]
        points_flow = pred_flow[:, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)].t()
        transformed_2d_points = points_2d + points_flow
        points_corr.append((points_2d, transformed_2d_points, points_3d))
    return points_corr

def get_2d_3d_corr_by_bw_flow(bw_flow, rendered_depths, ref_rotations, ref_translations, internel_k, valid_mask=None):
    '''
    Return a list of tuple, each element has three components, 
        2d points in the ref image, 2d points in the tgt image, and corresponding 3d points 
    '''
    num_images = len(bw_flow)
    warp_op = Warp(mode='bilinear', padding_mode='zeros')
    warped_depth = warp_op(rendered_depths[:, None], bw_flow).squeeze(dim=1)
    points_corr = []
    if valid_mask is None:
        valid_mask = torch.ones_like(rendered_depths).to(torch.bool)
    for i in range(num_images):
        tgt_2d_points_y, tgt_2d_points_x = torch.nonzero(valid_mask[i], as_tuple=True)
        tgt_2d_points_xy = torch.stack([tgt_2d_points_x, tgt_2d_points_y], dim=-1).to(torch.float32)
        points_flow = bw_flow[i, :, tgt_2d_points_y, tgt_2d_points_x].t()
        ref_2d_points = tgt_2d_points_xy + points_flow
        ref_points_depth = warped_depth[i, tgt_2d_points_y, tgt_2d_points_x]
        ref_points_xyz = torch.cat([ref_2d_points, torch.ones_like(tgt_2d_points_x)[:, None].to(torch.float32)], dim=-1)
        ref_points_xyz = ref_points_xyz * ref_points_depth[..., None]
        ref_points_xyz_camera_frame = torch.mm(torch.inverse(internel_k[i]), ref_points_xyz.t()).t()
        points_3d = torch.mm(torch.inverse(ref_rotations[i]), (ref_points_xyz_camera_frame - ref_translations[i, None]).t()).t()
        points_corr.append((ref_2d_points, tgt_2d_points_xy, points_3d))
    return points_corr


def get_3d_3d_corr_by_fw_flow(
    fw_flow:torch.Tensor, rendered_depths:torch.Tensor, real_depths:torch.Tensor, 
    ref_rotations:torch.Tensor, ref_translations:torch.Tensor,  internel_k:torch.Tensor, valid_mask=None):
    '''
    Return a list of tuple, each element has two components,
        3d points defined in camera frame, 3d points defined in object frame
    '''
    num_images = len(fw_flow)
    warp_op = Warp(mode='bilinear', padding_mode='zeros')
    warped_real_depth = warp_op(real_depths.unsqueeze(dim=1), fw_flow).squeeze(dim=1)
    if valid_mask is None:
        valid_mask = torch.ones_like(rendered_depths).to(torch.bool)
    points_corr = []
    for i in range(num_images):
        src_points_2d, points_3d_object_frame = cal_3d_2d_corr(
                rendered_depths[i], internel_k[i], ref_rotations[i], ref_translations[i], valid_mask[i])
        pred_flow = fw_flow[i]
        points_flow = pred_flow[:, src_points_2d[:, 1].to(torch.int64), src_points_2d[:, 0].to(torch.int64)].t()
        tgt_points_2d = src_points_2d + points_flow
        tgt_points_depth = warped_real_depth[i, src_points_2d[:, 1].to(torch.int64), src_points_2d[:, 0].to(torch.int64)]
        points_3d_camera_frame = lift_2d_to_3d(tgt_points_2d[:, 0], tgt_points_2d[:, 1], tgt_points_depth, internel_k[i])
        points_corr.append((points_3d_camera_frame, points_3d_object_frame))
    return points_corr


def get_3d_3d_corr_by_bw_flow(
    bw_flow:torch.Tensor, rendered_depths:torch.Tensor, real_depths:torch.Tensor,
    ref_rotations:torch.Tensor, ref_translations:torch.Tensor, internel_k:torch.Tensor, valid_mask=None):
    num_images = len(bw_flow)
    warp_op = Warp(mode='bilinear', padding_mode='zeros')
    warped_render_depth = warp_op(rendered_depths.unsqueeze(dim=1), bw_flow).squeeze(dim=1)
    points_corr = []
    if valid_mask is None:
        valid_mask = torch.ones_like(rendered_depths).to(torch.bool)
    for i in range(num_images):
        tgt_2d_points_y, tgt_2d_points_x = torch.nonzero(valid_mask[i], as_tuple=True)
        tgt_points_depth = real_depths[i, tgt_2d_points_y, tgt_2d_points_x]
        points_3d_camera_frame = lift_2d_to_3d(tgt_2d_points_x.float(), tgt_2d_points_y.float(), tgt_points_depth, internel_k[i])
        points_flow = bw_flow[i, :, tgt_2d_points_y, tgt_2d_points_x].t()
        ref_points_x, ref_points_y = tgt_2d_points_x + points_flow[:, 0], tgt_2d_points_y + points_flow[:, 1]
        ref_points_depth = warped_render_depth[i, tgt_2d_points_y, tgt_2d_points_x]
        _, points_3d_object_frame = lift_2d_to_3d(ref_points_x, ref_points_y, ref_points_depth, internel_k[i], ref_rotations[i], ref_translations[i])
        points_corr.append((points_3d_camera_frame, points_3d_object_frame))
    return points_corr

def get_3d_3d_corr_by_fw_flow_origspace(
    fw_flow:torch.Tensor, 
    rendered_depths:torch.Tensor, real_depths:torch.Tensor, 
    ref_rotations:torch.Tensor, ref_translations:torch.Tensor, 
    internel_k:torch.Tensor, ori_internel_k:torch.Tensor, 
    transform_matrixs:torch.Tensor, valid_mask=None):
    '''
    For points in the target image: extract the 2d points by flow, inverse them to original image space, interpolate depth, then lift to 3D camera frame.
    For points in the reference image: directly lift to 3D object frame.
    '''
    num_images = len(fw_flow)
    if valid_mask is None:
        valid_mask = torch.ones_like(rendered_depths).to(torch.bool)
    points_corr = []
    for i in range(num_images):
        src_points_2d, points_3d_object_frame = cal_3d_2d_corr(
                rendered_depths[i], internel_k[i], ref_rotations[i], ref_translations[i], valid_mask[i])
        pred_flow = fw_flow[i]
        points_flow = pred_flow[:, src_points_2d[:, 1].to(torch.int64), src_points_2d[:, 0].to(torch.int64)].t()
        tgt_points_2d = src_points_2d + points_flow
        tgt_points_2d = remap_points_to_origin_resolution(tgt_points_2d, transform_matrixs[i])
        tgt_points_depth = interpolate_depth(tgt_points_2d[:, 0], tgt_points_2d[:, 1], real_depths[i])
        points_3d_camera_frame = lift_2d_to_3d(tgt_points_2d[:, 0], tgt_points_2d[:, 1], tgt_points_depth, ori_internel_k[i])
        points_corr.append((points_3d_camera_frame, points_3d_object_frame))
    return points_corr


def get_3d_3d_corr_by_bw_flow_origspace(
    bw_flow:torch.Tensor, 
    rendered_depths:torch.Tensor, real_depths:torch.Tensor,
    ref_rotations:torch.Tensor, ref_translations:torch.Tensor, 
    internel_k:torch.Tensor, ori_internel_k:torch.Tensor, 
    transform_matrixs:torch.Tensor, valid_mask=None):
    '''
    For points in the target image: extract the valid 2d points by mask, inverse them to the original image space, interploate depth, then lift to 3D camera frame.
    For points in the reference image: same as getting points in the cropped image.
    '''
    num_images = len(bw_flow)
    warp_op = Warp(mode='bilinear', padding_mode='zeros')
    warped_render_depth = warp_op(rendered_depths.unsqueeze(dim=1), bw_flow).squeeze(dim=1)
    points_corr = []
    if valid_mask is None:
        valid_mask = torch.ones_like(rendered_depths).to(torch.bool)
    for i in range(num_images):
        tgt_2d_points_y, tgt_2d_points_x = torch.nonzero(valid_mask[i], as_tuple=True)
        mapped_tgt_2d_points = remap_points_to_origin_resolution(torch.stack([tgt_2d_points_x, tgt_2d_points_y], dim=-1).to(torch.float32), transform_matrixs[i])
        mapped_tgt_2d_points_x, mapped_tgt_2d_points_y = mapped_tgt_2d_points[:, 0], mapped_tgt_2d_points[:, 1]
        tgt_points_depth = interpolate_depth(mapped_tgt_2d_points_x, mapped_tgt_2d_points_y, real_depths[i])
        points_3d_camera_frame = lift_2d_to_3d(mapped_tgt_2d_points_x, mapped_tgt_2d_points_y, tgt_points_depth, ori_internel_k[i])
        points_flow = bw_flow[i, :, tgt_2d_points_y, tgt_2d_points_x].t()
        ref_points_x, ref_points_y = tgt_2d_points_x + points_flow[:, 0], tgt_2d_points_y + points_flow[:, 1]
        ref_points_depth = warped_render_depth[i, tgt_2d_points_y, tgt_2d_points_x]
        _, points_3d_object_frame = lift_2d_to_3d(ref_points_x, ref_points_y, ref_points_depth, internel_k[i], ref_rotations[i], ref_translations[i])
        points_corr.append((points_3d_camera_frame, points_3d_object_frame))
    return points_corr


def solve_pose_by_pnp(points_2d:torch.Tensor, points_3d:torch.Tensor, internel_k:torch.Tensor, **kwargs):
    '''
    Args:
        points_2d (Tensor): xy coordinates of 2d points, shape (N, 2)
        points_3d (Tenosr): xyz coordinates of 3d points, shape (N, 3)
        internel_k (Tensor): camera intrinsic, shape (3, 3)
        kwargs (dict):
    '''
    if points_2d.size(0) < 4:
        return None, None, False
    if kwargs.get('solve_pose_mode', 'ransacpnp') == 'ransacpnp':
        ransacpnp_parameter = kwargs.get('solve_pose_param', {})
        reprojectionError = ransacpnp_parameter.get('reprojectionerror', 3.0)
        iterationscount = ransacpnp_parameter.get('iterationscount', 100)
        retval, rotation_pred, translation_pred, inliers = cv2.solvePnPRansac(
            points_3d.cpu().numpy(), 
            points_2d.cpu().numpy(),
            internel_k.cpu().numpy(),
            None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=reprojectionError, iterationsCount=iterationscount
        )
        rotation_pred = cv2.Rodrigues(rotation_pred)[0].reshape(3, 3)
    elif kwargs.get('solve_pose_mode', 'ransacpnp') == 'progressive-x': 
        import pyprogressivex
        pose_ests, _ = pyprogressivex.find6DPoses(
            x1y1 = points_2d.cpu().data.numpy().astype(np.float64),
            x2y2z2 = points_3d.cpu().data.numpy().astype(np.float64),
            K = internel_k.cpu().numpy().astype(np.float64),
            threshold = 2,  
            neighborhood_ball_radius=20,
            spatial_coherence_weight=0.1,
            maximum_tanimoto_similarity=0.9,
            max_iters=400,
            minimum_point_number=6,
            maximum_model_number=1)
        if pose_ests.shape[0] == 0:
            retval = False
        else:
            retval = True
            rotation_pred = pose_ests[0:3, :3]
            translation_pred = pose_ests[0:3, 3]
    else:
        raise RuntimeError(f"Not supported pnp solver :{kwargs.get('solve_pose_mode')}")
    if retval:
        translation_pred = translation_pred.reshape(-1)
        if np.isnan(rotation_pred.sum()) or np.isnan(translation_pred.sum()):
            retval = False
    return rotation_pred, translation_pred, retval

def remap_points_to_origin_resolution(points_2d:torch.Tensor, transform_matrix:torch.Tensor):
    '''
    Remap 2d points on crop and resized patch to original image.
    '''
    num_points = len(points_2d)
    homo_points_2d = torch.cat([points_2d, points_2d.new_ones(size=(num_points, 1))], dim=-1)
    inverse_transform_matrix = torch.linalg.inv(transform_matrix)
    remapped_points_2d = torch.matmul(inverse_transform_matrix[:2, :], homo_points_2d.transpose(0, 1)).transpose(0, 1)
    return remapped_points_2d




def remap_pose_to_origin_resoluaion(pred_rotations_list, pred_translations_list, internel_k_list, meta_info_list):
    '''
    Remap pose predictions to original image resolution for all the objects in an image.
    As we perform different kinds of camera calibration, the remapped pose should follow the same way.
    '''
    num_images = len(pred_rotations_list)
    assert len(pred_translations_list) == len(internel_k_list) == len(meta_info_list) == num_images
    remapped_pred_rotations_list, remapped_pred_translations_list = [], []
    for j in range(num_images):
        pred_rotations, pred_translations = pred_rotations_list[j], pred_translations_list[j]
        internel_k, meta_info = internel_k_list[j], meta_info_list[j]
        if meta_info['geometry_transform_mode'] == 'adapt_intrinsic':
            remapped_pred_rotations_list.append(pred_rotations)
            remapped_pred_translations_list.append(pred_translations)
        else:
            transform_matrixs = meta_info['transform_matrix']
            inverse_transform_matrixs = np.linalg.inv(transform_matrixs)
            keypoints_3d = meta_info['keypoints_3d']
            pre_obj_num = len(pred_rotations)
            pred_rotations_np = pred_rotations.cpu().data.numpy()
            pred_translations_np = pred_translations.cpu().data.numpy()
            internel_k_np = internel_k.cpu().data.numpy()
            remapped_rotations, remapped_translations = [], []
            if meta_info['geometry_transform_mode'] == 'target_intrinsic':
                # 3*3 but not N*3*3, because the ori_k is for the whole image
                ori_k = meta_info['ori_k'] 
                for i in range(pre_obj_num):
                    remapped_rotation, remapped_translation, diff_pixel = remap_pose(
                        internel_k_np[i], pred_rotations_np[i], pred_translations_np[i], keypoints_3d[i], ori_k, inverse_transform_matrixs[i]
                    )
                    remapped_rotations.append(remapped_rotation)
                    remapped_translations.append(remapped_translation)
            elif meta_info['geometry_transform_mode'] == 'keep_intrinsic':
                for i in range(pre_obj_num):
                    remapped_rotation, remapped_translation, diff_pixel = remap_pose(
                        internel_k_np[i], pred_rotations_np[i], pred_translations_np[i], keypoints_3d[i], internel_k_np[i], inverse_transform_matrixs[i]
                    )
                    remapped_rotations.append(remapped_rotation)
                    remapped_translations.append(remapped_translation)
            else:
                raise RuntimeError
            remapped_rotations = torch.from_numpy(np.stack(remapped_rotations, axis=0)).to(torch.float32).to(pred_translations.device)
            remapped_translations = torch.from_numpy(np.stack(remapped_translations, axis=0)).to(torch.float32).to(pred_rotations.device)
            remapped_pred_translations_list.append(remapped_translations)
            remapped_pred_rotations_list.append(remapped_rotations)
    return remapped_pred_rotations_list, remapped_pred_translations_list




def depth_refine(
    depth_real:torch.Tensor, depth_render:torch.Tensor, mask_pred:torch.Tensor, 
    ref_translation:torch.Tensor, internel_k:torch.Tensor, mask_thr:float=0.8):
    # https://github.com/rasmushaugaard/surfemb/blob/master/surfemb/scripts/infer_refine_depth.py
    mask_depth = depth_real > 0
    mask_render = depth_render > 0
    mask_pred = mask_pred * mask_depth * mask_render
    total_mask = mask_pred > (mask_pred.max() * mask_thr)
    points_y, points_x = torch.nonzero(total_mask, as_tuple=True)
    depth_diff = depth_real[points_y, points_x] - depth_render[points_y, points_x]
    depth_adjustment = torch.median(depth_diff)

    h, w = depth_render.shape
    coords_y, coords_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords_xy = torch.stack([coords_x, coords_y], axis=-1).to(mask_pred.device)
    xy_ray_2d = torch.sum(coords_xy * mask_pred[..., None], dim=(0, 1))/mask_pred.sum()
    ray_3d = torch.matmul(torch.linalg.inv(internel_k), torch.cat([xy_ray_2d, xy_ray_2d.new_ones((1, ))], dim=0))
    ray_3d = ray_3d / ray_3d[-1]
    refined_translation = ref_translation + ray_3d * depth_adjustment
    return refined_translation

def get_flow_from_other_view(
    src_rotation:torch.Tensor, src_translation:torch.Tensor, 
    src_depth:torch.Tensor, src_mask:torch.Tensor,
    tgt_rotations:torch.Tensor, tgt_translations:torch.Tensor, 
    tgt_depth:torch.Tensor, tgt_mask:torch.Tensor,
    internel_k:torch.Tensor, other_view_flow:torch.Tensor, invalid_num=400, 
    filter_flow_by_depth=False, thr=0.2):
    '''
    Args:
        src_rotation: (3, 3)
        src_translation: (3, 3)
        src_depth: (H, W)
        tgt_rotations: (V, 3, 3)
        tgt_translations: (V, 3)
        tgt_depth: (V, H, W)
        internel_k: (3, 3)
        other_view_flow: (V, 2, H, W)
    '''
    def coords_grid(flow: torch.Tensor):
        B, _, H, W = flow.shape
        xx = torch.arange(0, W, device=flow.device, requires_grad=False)
        yy = torch.arange(0, H, device=flow.device, requires_grad=False)
        coords = torch.meshgrid(yy, xx)
        coords = torch.stack(coords[::-1], dim=0).float()
        grid = coords[None].repeat(B, 1, 1, 1) + flow
        return grid
    H, W = src_depth.shape
    view_num = tgt_rotations.size(1)
    other_view_coords = coords_grid(other_view_flow)
    view_flow = get_flow_from_delta_pose_and_depth(
        src_rotation[None].expand(view_num, -1, -1), 
        src_translation[None].expand(view_num, -1), 
        tgt_rotations, tgt_translations, 
        src_depth[None].expand(view_num, -1, -1),
        internel_k[None].expand(view_num, -1, -1), 
        invalid_num)
    
    view_flow = filter_flow_by_mask(view_flow, tgt_mask, invalid_num)
    if filter_flow_by_depth:
        view_flow = filter_flow_by_depth(
            view_flow, tgt_depth, src_depth[None].expand(view_num, -1, -1), invalid_num, thr)
    valid_mask = (view_flow[:, 0] < invalid_num) & (view_flow[:, 1] < invalid_num)
    warp_op = Warp()
    warped_other_view_coords = warp_op(other_view_coords, view_flow)
    dumpy_flow = torch.zeros((view_num, 2, H, W)).to(src_depth.device)
    dump_coords = coords_grid(dumpy_flow)
    return warped_other_view_coords - dump_coords, valid_mask

def solve_pose_by_ransac_kabsch(points_3d_camera_frame:torch.Tensor, points_3d_object_frame:torch.Tensor, **kwargs):
    min_sampels = kwargs.get('min_samples', 3)
    residual_threshold = kwargs.get('res_thresh', 20)
    max_trials = kwargs.get('max_trials', 200)    
    ransac = BatchRansacEstimator(min_samples=min_sampels, residual_threshold=residual_threshold, max_trials=max_trials)
    ret = ransac.fit(points_3d_object_frame, points_3d_camera_frame)    
    if ret['retval']:
        rotation_pred, translation_pred = ret['best_params'][0], ret['best_params'][1]
        return rotation_pred, translation_pred, True
    else:
        return None, None, False

def get_point_cloud_from_depth(depth, K):
    cam_fx, cam_fy, cam_cx, cam_cy = K[0,0], K[1,1], K[0,2], K[1,2]

    im_H, im_W = depth.shape
    xmap = torch.arange(im_W, dtype=torch.float32).unsqueeze(0).repeat(im_H, 1).to(depth)
    ymap = torch.arange(im_H, dtype=torch.float32).unsqueeze(1).repeat(1, im_W).to(depth)

    pt2 = depth.float()
    pt0 = (xmap - cam_cx) * pt2 / cam_fx
    pt1 = (ymap - cam_cy) * pt2 / cam_fy

    cloud = torch.stack([pt0, pt1, pt2], dim=2)
    return cloud

def depth_to_point_cloud_sample(gt_depths, rd_depths, intrinsics):
    sampled_gt_point_clouds = []
    sampled_rd_point_clouds = []
    grid_x, grid_y = torch.meshgrid(torch.arange(4, gt_depths.size(-1), 8), torch.arange(4, gt_depths.size(-1), 8), indexing='ij')
    for i in range(len(gt_depths)):
        gt_depth = gt_depths[i] / 1000.0
        rd_depth = rd_depths[i] / 1000.0
        K = intrinsics[i]
        gt_point_cloud = get_point_cloud_from_depth(gt_depth, K)
        rd_point_cloud = get_point_cloud_from_depth(rd_depth, K)
        sampled_gt_point_cloud = gt_point_cloud[grid_x, grid_y].view(-1, 3)     # (N, 3)
        sampled_rd_point_cloud = rd_point_cloud[grid_x, grid_y].view(-1, 3)     # (N, 3)
        sampled_gt_point_clouds.append(sampled_gt_point_cloud)
        sampled_rd_point_clouds.append(sampled_rd_point_cloud)
    return torch.stack(sampled_gt_point_clouds, dim=0), torch.stack(sampled_rd_point_clouds, dim=0) # (B, N, 3)

def depth_to_point_cloud(gt_depths, rd_depths, intrinsics):
    gt_point_clouds = []
    rd_point_clouds = []
    for i in range(len(gt_depths)):
        gt_depth = gt_depths[i] / 1000.0
        rd_depth = rd_depths[i] / 1000.0
        K = intrinsics[i]
        gt_point_cloud = get_point_cloud_from_depth(gt_depth, K).view(-1, 3)
        rd_point_cloud = get_point_cloud_from_depth(rd_depth, K).view(-1, 3)
        gt_point_clouds.append(gt_point_cloud)
        rd_point_clouds.append(rd_point_cloud)
    return torch.stack(gt_point_clouds, dim=0), torch.stack(rd_point_clouds, dim=0) # (B, N, 3)