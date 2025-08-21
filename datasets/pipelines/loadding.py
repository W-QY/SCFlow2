import mmcv, cv2
import numpy as np
from bop_toolkit_lib.inout import load_depth
from .builder import PIPELINES
from ..mask import BitmapMasks
from ..bop_object_utils import get_point_cloud_from_depth
import os
import json
import trimesh
from .rendering_online import RendererOnline
import torch
import glob

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

@PIPELINES.register_module()
class ImageRendering:
    def __init__(self, device_id, mean, std, renderer, rgb_to_gary=False):
        self.renderer = renderer
        self.device_id = device_id
        self.normalize_mean = mean
        self.normalize_std = std
        self.rgb_to_gary = rgb_to_gary
    def __call__(self, results):
        if self.device_id == -1:
            self.device_id = torch.cuda.current_device()
        device = torch.device(f'cuda:{self.device_id}')
        renderer = RendererOnline(**self.renderer)
        renderer.to(device)
        ref_rotations, ref_translations = torch.tensor(results['ref_rotations']).to(device), torch.tensor(results['ref_translations']).to(device)
        internel_k = torch.tensor(results['k']).to(device)
        paths, mesh_scale_data= results['mesh_obj_paths'], results['mesh_scale_data']
        render_outputs = renderer(ref_rotations, ref_translations, internel_k, paths, mesh_scale_data)
        if render_outputs == None:
            return None
        rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
        rendered_depths = rendered_fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        rendered_masks = (rendered_depths > 0).to(torch.float32)
        if self.rgb_to_gary:
            rendered_images = rgb_to_gray(rendered_images)
        for i in range(len(rendered_masks)):
            non_zero_count = torch.sum(rendered_masks[i] != 0)
            if non_zero_count < 1024 or torch.all(torch.isclose(rendered_images[i], torch.tensor(0.5))):
                return None
        normalize_mean, normalize_std = self.normalize_mean, self.normalize_std
        normalize_mean = torch.Tensor(normalize_mean).view(1, 3, 1, 1).to(device) / 255.
        normalize_std = torch.Tensor(normalize_std).view(1, 3, 1, 1).to(device) / 255.
        rendered_images = (rendered_images - normalize_mean)/normalize_std
        results['rendered_depths'] = rendered_depths.cpu().numpy()
        results['rendered_images'] = rendered_images.cpu().numpy()
        results['rendered_masks'] = rendered_masks.cpu().numpy()
        torch.cuda.set_device(f'cuda:{self.device_id}')
        torch.cuda.empty_cache()
        return results

@PIPELINES.register_module()
class LoadImages:
    def __init__(self, 
                color_type='color',
                to_float32=False,
                resize=None,
                file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.resize = resize
    
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filepath = results['img_path']
        img_bytes = self.file_client.get(filepath=filepath)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if len(img.shape) == 2: # if gray, then dumplicate 3
            img = np.stack([img] * 3, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        # if self.resize:
        #     img = cv2.resize(img, (int(img.shape[1] * self.resize), int(img.shape[0] * self.resize)))
        #     results['k'][:, :2, :] *= self.resize
            # results['ref_translations'] *= self.resize
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

@PIPELINES.register_module()
class LoadDepth:
    def __init__(self):
        pass
    
    def __call__(self, results):
        filepath = results['depth_path']
        results['depths'] = load_depth(filepath)
        return results


def rle_to_binary_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')

    start = 0
    for i in range(len(counts)-1):
        start += counts[i] 
        end = start + counts[i+1] 
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask



@PIPELINES.register_module()
class LoadMasks:
    def __init__(self,
                binarize=True,
                merge=False,
                load_full_mask=False,
                ref_mask_dir=None,
                file_client_args=dict(backend='disk'),
                eps=1e-5):
        self.binarize = binarize
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.eps = eps
        self.merge = merge
        self.load_full_mask = load_full_mask
        self.ref_mask_dir = ref_mask_dir

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        mask_paths = results['gt_mask_path']
        height, width, _ = results['img_shape']

        masks = []
        if self.load_full_mask:  # used for loading masks without gt_1abel
            mask_dir = mask_paths[0].rsplit('/', 1)[0]
            image_id = mask_paths[0].rsplit('/', 1)[1].rsplit("_000000")[0]
            paths = sorted(glob.glob(os.path.join(mask_dir, f'{image_id}*.png')))
            merged_mask = np.zeros((height, width), dtype=np.uint8)
            for path in paths:
                img_bytes = self.file_client.get(filepath=path)
                mask = mmcv.imfrombytes(img_bytes, flag='unchanged')
                if self.binarize:
                    dtype = mask.dtype
                    if mask.max() < self.eps:
                        mask[...] = 0
                    else:
                        mask = (mask / mask.max()).astype(dtype)
                    masks.append(mask)
            #     merged_mask = np.maximum(merged_mask, mask)
            # masks = [merged_mask.copy() for _ in range(len(results['labels']))]
        elif self.ref_mask_dir:
            masks_info = results['ref_mask_info']
            for mask_info in masks_info:
                mask = rle_to_binary_mask(mask_info['segmentation'])
                mask = mask.astype(int)
                masks.append(mask)
        else:
            for path in mask_paths:
                img_bytes = self.file_client.get(filepath=path)
                mask = mmcv.imfrombytes(img_bytes, flag='unchanged')
                if self.binarize:
                    dtype = mask.dtype
                    if mask.max() < self.eps:
                        mask[...] = 0
                    else:
                        mask = (mask / mask.max()).astype(dtype)
                masks.append(mask)
        mask = BitmapMasks(masks, height, width)
        results['gt_masks'] = mask
        return results


@PIPELINES.register_module()
class LoadMasksGSO:
    def __init__(self,
                binarize=True,
                merge=False,
                dilate_mask=False,
                aug_mask=False, 
                file_client_args=dict(backend='disk'),
                eps=1e-5):
        self.binarize = binarize
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.eps = eps
        self.merge = merge
        self.dilate_mask = dilate_mask
        self.aug_mask = aug_mask
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        mask_paths = results['gt_mask_path']
        height, width, _ = results['img_shape']
        obj_index_in_image = results['obj_index_in_image']

        masks = []
        for i in range(len(mask_paths)):
            path = mask_paths[i]
            index = obj_index_in_image[i]
            mask_rle = json.load(open(os.path.join(path), 'rb'))
            mask_rle = {int(k): v for k, v in mask_rle.items()}
            try:
                mask = rle_to_binary_mask(mask_rle[index])
            except:
                print(path)
                return None
            mask = mask.astype(int)
            if self.dilate_mask and np.random.rand() < 0.5:
                mask = np.array(mask>0).astype(np.uint8)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)), iterations=4)
            if self.aug_mask and np.random.rand() < 0.6:
                if np.random.rand() < 0.5:
                    mask = np.array(mask>0).astype(np.uint8)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)), iterations=4)
                else:
                    mask = np.array(mask>0).astype(np.uint8)
                    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)

            masks.append(mask)
        
        mask = BitmapMasks(masks, height, width)
        results['gt_masks'] = mask
        return results


@PIPELINES.register_module()
class LoadModelEval:
    def __init__(self, depth_sample_num=1024, mesh_eval_sample=4096):
        self.depth_sample_num = depth_sample_num
        self.mesh_eval_sample = mesh_eval_sample
    def __call__(self, results):
        model_list = []
        model_eval_list = []
        mesh_paths = results['mesh_ply_paths']
        mesh_scales = results['mesh_scale_data']
        for i in range(len(mesh_paths)):
            model_path = mesh_paths[i]
            mesh = trimesh.load(model_path)
            if not mesh_scales:     # empty: GSO, ShapeNet
                mess_eval = mesh.vertices.view(np.ndarray).astype(np.float32)
            else:                   # Objaverse
                mess_eval = mesh.vertices.view(np.ndarray).astype(np.float32) * mesh_scales[i]         
            if mess_eval.shape[0] >= self.mesh_eval_sample:
                indices = np.random.choice(mess_eval.shape[0], self.mesh_eval_sample, replace=False)
            else:
                indices = np.random.choice(mess_eval.shape[0], self.mesh_eval_sample, replace=True)
            mess_eval = mess_eval[indices]
            model_eval_list.append(np.array(mess_eval))
            # model_list.append(np.array(model_points))
            del mesh
        # results['model_list'] = np.array(model_list)
        results['model_eval_list'] = np.array(model_eval_list)
        return results

@PIPELINES.register_module()
class GetPointCloudGSO:
    def __init__(self, filter_point_cloud=True, minimum_points=256, depth_sample_num=1024, mesh_eval_sample=4096, filter_depth=True, filter_rgb=False):
        self.filter_point_cloud = filter_point_cloud
        self.minimum_points = minimum_points
        self.depth_sample_num = depth_sample_num
        self.filter_depth = filter_depth
        self.filter_rgb = filter_rgb
        self.mesh_eval_sample = mesh_eval_sample
    #-# origin call:

    def __call__(self, results):

        model_list = []
        model_eval_list = []
        mesh_paths = results['mesh_ply_paths']
        mesh_scales = results['mesh_scale_data']
        for i in range(len(mesh_paths)):
            model_path = mesh_paths[i]
            mesh = trimesh.load(model_path)
            if not mesh_scales:     # empty
                model_points = mesh.sample(self.depth_sample_num).astype(np.float32) / 1000.0 
                mess_eval = mesh.vertices.view(np.ndarray).astype(np.float32)
            else:
                model_points = mesh.sample(self.depth_sample_num).astype(np.float32) / 1000.0 * mesh_scales[i]
                mess_eval = mesh.vertices.view(np.ndarray).astype(np.float32) * mesh_scales[i]         
            if mess_eval.shape[0] >= self.mesh_eval_sample:
                indices = np.random.choice(mess_eval.shape[0], self.mesh_eval_sample, replace=False)
            else:
                indices = np.random.choice(mess_eval.shape[0], self.mesh_eval_sample, replace=True)
            mess_eval = mess_eval[indices]
            model_eval_list.append(np.array(mess_eval))
            model_list.append(np.array(model_points))
            del mesh
        results['model_list'] = np.array(model_list)
        results['model_eval_list'] = np.array(model_eval_list)

        gt_depths = results.get('depths')
        gt_masks = results['gt_masks']
        images = results['img']
        assert len(model_list) == len(gt_depths)
        intrinsics = results.get('k')
        cloud_list = []
        for i in range(len(model_list)):
            depth = gt_depths[i] / 1000.0
            model = model_list[i]
            mask = gt_masks[i].masks    #  * (depth > 0)
            if self.filter_depth:
                gt_depths[i][np.squeeze(mask, axis=0) == 0] = 0
            if self.filter_rgb:
                images[i][np.squeeze(mask, axis=0) == 0] = 128
            K = intrinsics[i]
            choose = mask.astype(np.float32).flatten().nonzero()[0]
            if len(choose) < self.minimum_points:
                return None
            cloud = get_point_cloud_from_depth(depth, K)
            cloud = cloud.reshape(-1, 3)[choose, :]
            if len(choose) <= self.depth_sample_num:
                choose_idx = np.random.choice(np.arange(len(choose)), self.depth_sample_num)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.depth_sample_num, replace=False)
            choose = choose[choose_idx]
            cloud = cloud[choose_idx]
            cloud_list.append(cloud)
        results['cloud_list'] = np.array(cloud_list)
        results['depths'] = np.array(gt_depths)
        results['img'] = images
        return results

@PIPELINES.register_module()
class GetPointCloud:
    def __init__(self, filter_point_cloud=True, minimum_points=256, depth_sample_num=1024, filter_depth=True, filter_rgb=False):
        self.filter_point_cloud = filter_point_cloud
        self.minimum_points = minimum_points
        self.depth_sample_num = depth_sample_num
        self.filter_depth = filter_depth
        self.filter_rgb = filter_rgb
    #-# origin call:
    # if True:
    def __call__(self, results):
        model_list = results.get('model_list')
        model_diameter_list = results.get('model_diameter_list')
        gt_depths = results.get('depths')
        gt_masks = results['gt_masks']
        images = results['img']
        assert len(model_list) == len(gt_depths)
        intrinsics = results.get('k')
        cloud_list = []
        for i in range(len(model_list)):
            depth = gt_depths[i] / 1000.0
            model = model_list[i]
            mask = gt_masks[i].masks    #  * (depth > 0)
            if self.filter_depth:
                gt_depths[i][np.squeeze(mask, axis=0) == 0] = 0
            if self.filter_rgb:
                images[i][np.squeeze(mask, axis=0) == 0] = 128
            K = intrinsics[i]
            model_diameter = model_diameter_list[i]
            cloud = get_point_cloud_from_depth(depth, K).reshape(-1, 3)
            choose = mask.astype(np.float32).flatten().nonzero()[0]
            if len(choose) < self.minimum_points:
                if self.filter_point_cloud:
                    return None
                else:
                    choose = np.array([i for i in range(len(cloud))])
            cloud = cloud[choose, :]
            # center = np.mean(cloud, axis=0)
            # tmp_cloud = cloud - center[None, :]
            # flag = np.linalg.norm(tmp_cloud, axis=1) < model_diameter * 0.6
            # if np.sum(flag) < self.minimum_points:
            #     return None
            # choose = choose[flag]
            # cloud = cloud[flag]
            if len(choose) <= self.depth_sample_num:
                choose_idx = np.random.choice(np.arange(len(choose)), self.depth_sample_num)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.depth_sample_num, replace=False)
            choose = choose[choose_idx]
            cloud = cloud[choose_idx]
            cloud_list.append(cloud)
        results['img'] = images
        results['cloud_list'] = cloud_list 
        results['depths'] = gt_depths
        return results

    #-# changed call
    # else:
    #     def __call__(self, results):
    #         model_list = results.get('model_list')
    #         model_diameter_list = results.get('model_diameter_list')
    #         gt_depths = results.get('depths')
    #         gt_masks = results['gt_masks']
    #         assert len(model_list) == len(gt_depths)
    #         intrinsics = results.get('k')
    #         cloud_list = []
    #         for i in range(len(model_list)):
    #             depth = gt_depths[i] / 1000.0
    #             model = model_list[i]
    #             mask = np.squeeze(gt_masks[i].masks, axis=0)
    #             depth[mask == 0] = 0
    #             K = intrinsics[i]
    #             point_cloud = get_point_cloud_from_depth(depth, K)
    #             grid_x, grid_y = np.meshgrid(np.arange(4, 256, 8), np.arange(4, 256, 8))
    #             sampled_point_cloud = point_cloud[grid_x.ravel(), grid_y.ravel()]
    #             cloud_list.append(sampled_point_cloud)
    #         results['cloud_list'] = cloud_list 
    #         return results
