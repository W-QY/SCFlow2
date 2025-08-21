import mmcv
from mmcv.parallel import DataContainer
from mmcv.utils import build_from_cfg
import numpy as np
import transforms3d
from os import path as osp
import trimesh
from glob import glob
from .builder import PIPELINES
from ..utils import to_tensor
from ..pose import project_3d_point, eval_rot_error, eval_tran_error, load_mesh
from ..mask import BitmapMasks

from copy import deepcopy
import cv2
from scipy import interpolate

@PIPELINES.register_module()
class DepthLinInterp:
    """
    Linear interpolation of depth maps
    """
    def __init__(self, clipmin=10.0):
        self.clipmin = clipmin
    def lin_interp(self, depth):
        x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                        np.arange(depth.shape[0]).astype("float32"))
        xx = x[depth > 0]
        yy = y[depth > 0]
        zz = depth[depth > 0]
        grid = interpolate.griddata((xx, yy), zz.ravel(),
                                    (x, y), method='nearest')
        grid = np.where(grid < self.clipmin, self.clipmin, grid)
        return grid

    def __call__(self, results):
        depths = results['depths']
        lin_depths = []
        for depth in depths:
            lin_depths.append(self.lin_interp(depth))
        results['depths'] = lin_depths
        return results

@PIPELINES.register_module()
class MatchMask:
    '''
    Compute the bbox for the jittered pose, aka reference pose, add 'ref_bboxes' key
    '''
    def __init__(self, 
                bbox_field='ref_bboxes'
                ):
        self.bbox_field = bbox_field
    def calculate_iou(self, bbox1, bbox2):    # bbox: (x y w h)
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        w1, h1 = x2_1 - x1_1, y2_1 - y1_1
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2
        inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
        inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area

    def __call__(self, results):
        height, width, _ = results['img_shape']
        labels = results['labels']
        ref_bboxes = results[self.bbox_field]
        masks = results['gt_masks']     # x1, y1, x2, y2
        obj_num = len(labels)
        mask_bboxes = []
        for k in range(len(masks)):
            mask = masks[k].masks
            non_zero_pixels = cv2.findNonZero(mask[0])
            x, y, w, h = cv2.boundingRect(non_zero_pixels)
            mask_bboxes.append([x, y, x + w, y + h])
        return_mask = []
        for i in range(obj_num):
            iou_list = []
            for j in range(len(mask_bboxes)):
                iou_list.append(self.calculate_iou(ref_bboxes[i], mask_bboxes[j]))
            mask_index = iou_list.index(max(iou_list))
            return_mask.append(masks[mask_index])
        return_mask = BitmapMasks(return_mask, height, width)
        results['gt_masks'] = return_mask
        return results


@PIPELINES.register_module()
class ProjectKeypoints:
    '''
    Project 3D points to 2D image plane, add 'keypoints_2d' key and 'keypoints_3d_camera_frame' key
    '''
    def __init__(self, 
                clip_border=False):
        self.clip_border = clip_border

    def __call__(self, results):
        # this is 3d keypoints defined on object space
        keypoints_3d = results['gt_keypoints_3d']
        translations, rotations = results['gt_translations'], results['gt_rotations']
        num_obj = len(translations)
        k = results['k']
        keypoints_2d, keypoints_3d_camera_frame = [], []
        for i in range(num_obj):
            keypoint_3d = keypoints_3d[i]
            rotation, translation = rotations[i], translations[i]
            k_i = k[i]
            keypoint_2d, keypoint_3d_camera_frame = project_3d_point(keypoint_3d, k_i, rotation, translation, return_3d=True)
            keypoints_2d.append(keypoint_2d)
            keypoints_3d_camera_frame.append(keypoint_3d_camera_frame)
        keypoints_2d = np.stack(keypoints_2d, axis=0).astype(np.float32)
        keypoints_3d_camera_frame = np.stack(keypoints_3d_camera_frame, axis=0).astype(np.float32)
        results['gt_keypoints_3d_camera'] = keypoints_3d_camera_frame
        results['gt_keypoints_2d'] = keypoints_2d
        return results

@PIPELINES.register_module()
class ComputeBbox:
    '''
    Compute the bbox for the jittered pose, aka reference pose, add 'ref_bboxes' key
    '''
    def __init__(self, 
                mesh_dir, 
                clip_border=True, 
                filter_invalid=True,
                pose_field=['ref_rotations', 'ref_translations'], 
                bbox_field='ref_bboxes'):
        self.mesh_dir = mesh_dir
        self.meshes = load_mesh(mesh_dir)
        mesh_vertices = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
        self.mesh_vertices = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in mesh_vertices]
        self.clip_border = clip_border
        self.filter_invalid = filter_invalid
        self.pose_field = pose_field
        self.bbox_field = bbox_field


    def __call__(self, results):
        labels = results['labels']
        ref_rotations, ref_translations = results[self.pose_field[0]], results[self.pose_field[1]]
        ks = results['k']
        obj_num = len(labels)
        bboxes = []
        for i in range(obj_num):
            ref_rotation, ref_translation = ref_rotations[i], ref_translations[i]
            label, k = labels[i], ks[i]
            points_2d = project_3d_point(self.mesh_vertices[label], k, ref_rotation, ref_translation)
            points_x, points_y = points_2d[:, 0], points_2d[:, 1]
            left, right = points_x.min(), points_x.max()
            top, bottom = points_y.min(), points_y.max()
            bbox = np.array([left, top, right, bottom], dtype=np.float32)
            bboxes.append(bbox)
        if obj_num > 0:
            bboxes = np.stack(bboxes, axis=0)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
        if self.clip_border:
            height, width, _ = results['img'].shape
            left, right = np.clip(bboxes[:, 0], a_min=0, a_max=width), np.clip(bboxes[:, 2], a_min=0, a_max=width)
            top, bottom = np.clip(bboxes[:, 1], a_min=0, a_max=height), np.clip(bboxes[:, 3], a_min=0, a_max=height)
            bboxes = np.stack([left, top, right, bottom], axis=1)
        if self.filter_invalid:
            ori_shape = results['ori_shape']
            if np.sum((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) > ori_shape[0] * ori_shape[1]):
                return None
        results[self.bbox_field] = bboxes
        return results


@PIPELINES.register_module()
class ComputeBboxGSO:
    '''
    Compute the bbox for the jittered pose, aka reference pose, add 'ref_bboxes' key
    '''
    def __init__(self, 
                # mesh_dir, 
                # clip_border=True, 
                # filter_invalid=True,
                # pose_field=['ref_rotations', 'ref_translations'], 
                bbox_field='ref_bboxes'):
        # self.mesh_dir = mesh_dir
        # self.meshes = load_mesh(mesh_dir)
        # mesh_vertices = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
        # self.mesh_vertices = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in mesh_vertices]
        # self.clip_border = clip_border
        # self.filter_invalid = filter_invalid
        # self.pose_field = pose_field
        self.bbox_field = bbox_field


    def __call__(self, results):
        # labels = results['labels']
        # ref_rotations, ref_translations = results[self.pose_field[0]], results[self.pose_field[1]]
        # ks = results['k']
        # obj_num = len(labels)
        # bboxes = []
        # for i in range(obj_num):
        #     ref_rotation, ref_translation = ref_rotations[i], ref_translations[i]
        #     label, k = labels[i], ks[i]
        #     points_2d = project_3d_point(self.mesh_vertices[label], k, ref_rotation, ref_translation)
        #     points_x, points_y = points_2d[:, 0], points_2d[:, 1]
        #     left, right = points_x.min(), points_x.max()
        #     top, bottom = points_y.min(), points_y.max()
        #     bbox = np.array([left, top, right, bottom], dtype=np.float32)
        #     bboxes.append(bbox)
        # if obj_num > 0:
        #     bboxes = np.stack(bboxes, axis=0)
        # else:
        #     bboxes = np.zeros((0, 4), dtype=np.float32)
        # if self.clip_border:
        #     height, width, _ = results['img'].shape
        #     left, right = np.clip(bboxes[:, 0], a_min=0, a_max=width), np.clip(bboxes[:, 2], a_min=0, a_max=width)
        #     top, bottom = np.clip(bbox[:, 1], a_min=0, a_max=height), np.clip(bbox[:, 3], a_min=0, a_max=height)
        #     bbox = np.stack([left, top, right, bottom], axis=1)
        # if self.filter_invalid:
        #     ori_shape = results['ori_shape']
        #     if np.sum((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) > ori_shape[0] * ori_shape[1]):
        #         return None
        results[self.bbox_field] = results['gt_bboxes']
        return results



@PIPELINES.register_module()
class HandleSymmetry:
    '''
    Symmetry objects handling.
    If an object is symmetry on one axis, then the pose may have multiple resonable types.
    We will compute all the possible pose choices and add the possible pose choices to the pipeline results 
        as new pose annotations.
    '''
    def __init__(self, symmetry_types=None):
        self.symmetry_types = symmetry_types

    @classmethod
    def handle_symmetry(cls, rotations, labels, mesh_symmetry_types):
        assert len(labels) == len(rotations)
        axes_mapping = {'x':'sxyz', 'y':'syxz', 'z':'szyx'}
        for symmetry_label in mesh_symmetry_types:
            # Note: label is 0-based, while label in mesh_symmetry_types is 1-based.
            mask = (labels == int(symmetry_label.split('_')[-1])-1)
            if mask.sum() > 0:
                symmetry = mesh_symmetry_types[symmetry_label]
                rotations_l = rotations[mask]
                for axis in symmetry:
                    mod = symmetry[axis] * np.pi / 180
                    axes = axes_mapping[axis]
                    for i, rotation in enumerate(rotations_l):
                        old_ai, aj, ak = transforms3d.euler.mat2euler(rotation, axes=axes)
                        new_ai = 0 if mod == 0 else np.fmod(old_ai, mod)
                        rotation = transforms3d.euler.euler2mat(new_ai, aj, ak, axes=axes)
                        rotations_l[i] = rotation
                rotations[mask] = rotations_l
        return rotations
    
    def __call__(self, results):
        label_field = results['label_fields'][0]
        labels = results.get(label_field)
        for pose_field in results.get('pose_fields'):
            rotations = results.get(pose_field[0]) 
            rotations = self.handle_symmetry(rotations, labels, self.symmetry_types)
            results[pose_field[0]] = rotations
        return results


@PIPELINES.register_module()
class HandleSymmetryV2:
    def __init__(self, info_path:str, only_continuous=False):
        self.only_continuous = only_continuous
        models_info = mmcv.load(info_path)
        models_sym_info = dict()

        for model in models_info:
            model_info = models_info[model]
            if 'symmetries_discrete' in model_info:
                trans_disc = [{'R':np.eye(3, 3), 't':np.zeros(3).reshape(3), 'R_inv':np.eye(3, 3)}]
                for sym in model_info['symmetries_discrete']:
                    sym_4x4 = np.reshape(sym, (4, 4))
                    R = sym_4x4[:3, :3]
                    t = sym_4x4[:3, 3].reshape(3)
                    R_inv = np.linalg.inv(R)
                    trans_disc.append({'R': R, 't': t, 'R_inv':R_inv})
                models_sym_info[model] = dict()
                models_sym_info[model]['discrete'] = trans_disc
            if 'symmetries_continuous' in model_info:
                trans_conti = []
                for sym in model_info['symmetries_continuous']:
                    axis = np.array(sym['axis'])
                    offset = np.array(sym['offset']).reshape(3)
                    trans_conti.append({'axis':axis, 'offset':offset})
                if model not in models_sym_info:
                    models_sym_info[model] = dict()
                models_sym_info[model]['continuous'] = trans_conti
        self.models_sym_info = models_sym_info

    @classmethod
    def handle_discrete_symmetry(cls, sym_info:dict, rotation:np.ndarray, translation:np.ndarray):
        if 'discrete' not in sym_info:
            return rotation, translation
        discrete_sym_info = sym_info['discrete']
        trans = np.stack([info['R_inv'] for info in discrete_sym_info], axis=0)
        offset = np.stack([info['t'] for info in discrete_sym_info], axis=0)
        residual = np.matmul(rotation, trans) - np.eye(3, 3)[None]
        residual = np.linalg.norm(residual, ord='fro', axis=(1, 2))
        best_trans = trans[np.argmin(residual)]
        best_offset = offset[np.argmin(residual)]
        mapped_rotation = np.matmul(rotation, best_trans)
        mapped_translation = translation - np.matmul(mapped_rotation, best_offset)
        return mapped_rotation, mapped_translation

    @classmethod
    def handle_continuous_symmetry(cls, sym_info:dict, rotation:np.ndarray, translation:np.ndarray):
        if 'continuous' not in sym_info:
            return rotation, translation
        continuous_sym_info = sym_info['continuous']
        assert len(continuous_sym_info) == 1, "only support continuous symmetry around one axis"
        r_11, r_12, r_13 = rotation[0, 0], rotation[0, 1], rotation[0, 2]
        r_21, r_22, r_23 = rotation[1, 0], rotation[1, 1], rotation[1, 2]
        r_31, r_32, r_33 = rotation[2, 0], rotation[2, 1], rotation[2, 2]
        continuous_sym_info = continuous_sym_info[0]
        if np.allclose(continuous_sym_info['axis'], np.array([0, 0, 1])):
            # z axis
            angle = np.arctan2(r_21-r_12, r_11+r_22)
            best_trans = transforms3d.euler.euler2mat(angle, 0, 0, axes='szyx')
        elif np.allclose(continuous_sym_info['axis'], np.array([1, 0, 0])):
            # x axis
            angle = np.arctan2(r_23-r_32, r_22+r_33) #noqa
            best_trans = transforms3d.euler.euler2mat(angle, 0, 0, axes='sxyz')
        else:
            # y axis
            angle = np.arctan2(r_13-r_31, r_11+r_33) #noqa
            best_trans = transforms3d.euler.euler2mat(angle, 0, 0, axes='syxz')
        mapped_rotation = rotation @ np.linalg.inv(best_trans)   
        return mapped_rotation, translation


    def __call__(self, results):
        label_field = results['label_fields'][0]
        labels = results.get(label_field)
        obj_num = len(labels)
        for pose_field in results.get('pose_fields'):
            rotations = results.get(pose_field[0]) 
            translations = results.get(pose_field[1])
            for i in range(obj_num):
                label, rotation, translation = labels[i], rotations[i], translations[i]
                if str(label+1) in self.models_sym_info:
                    sym_info = self.models_sym_info[str(label+1)]
                    rotation, translation = self.handle_continuous_symmetry(sym_info, rotation, translation)
                    if not self.only_continuous:
                        rotation, translation = self.handle_discrete_symmetry(sym_info, rotation, translation)
                    rotations[i] = rotation
                    translations[i] = translation
            results[pose_field[0]] = rotations
            results[pose_field[1]] = translations
        return results

    



@PIPELINES.register_module()
class FilterAnnotations:
    def __init__(self, angle_limit, translation_limit, add_limit, mesh_dir, mesh_diameter):
        self.angle_limit = angle_limit
        self.translation_limit = translation_limit
        self.add_limit = add_limit
        self.mesh_diameters = mesh_diameter
        self.meshes = load_mesh(mesh_dir)
        mesh_vertices = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
        self.mesh_vertices = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in mesh_vertices]



    def __call__(self, results):
        ref_rotations, ref_translations = results['ref_rotations'], results['ref_translations']
        gt_rotations, gt_translations = results['gt_rotations'], results['gt_translations']
        num_obj = len(gt_rotations)
        labels = results['labels']
        add_error = np.zeros((num_obj,), dtype=np.float32)
        for i in range(num_obj):
            verts = self.mesh_vertices[labels[i]]
            gt_points = (np.matmul(gt_rotations[i], verts.T) + gt_translations[i][:, None]).T
            ref_points = (np.matmul(ref_rotations[i], verts.T) + ref_translations[i][:, None]).T
            add_error[i] = np.linalg.norm(gt_points - ref_points, axis=-1).mean() / self.mesh_diameters[labels[i]]
        add_keep = add_error < self.add_limit
        if add_keep.sum() == 0:
            return None

        translation_error, depth_error, xy_error = eval_tran_error(gt_translations, ref_translations)
        translation_keep = translation_error < self.translation_limit
        if translation_keep.sum() == 0:
            return None

        rotation_error = eval_rot_error(gt_rotations, ref_rotations)
        rotation_keep = rotation_error < self.angle_limit
        if rotation_keep.sum() == 0:
            return None 
        
        keep = rotation_keep & translation_keep & add_keep
        if keep.sum() == 0:
            return None
        
        results['init_add_error'] = add_error
        results['init_rot_error'] = rotation_error
        results['init_trans_error'] = translation_error
        
        for key in results.get('annot_fields'):
            results[key] = results[key][keep]
        keep_images = []
        for i, image in enumerate(results['img']):
            if keep[i]:
                keep_images.append(image)
        results['img'] = keep_images
        return results
        
        


@PIPELINES.register_module()
class ToTensor:
    def __init__(self, stack_keys=['img']):
        self.stack_keys = stack_keys

    def __call__(self, results):
        # format image
        image_keys = results.get('image_filelds', ['img'])
        for key in image_keys:
            img = results[key]
            if isinstance(img, (list, tuple)):
                # multiple patches
                ndim = img[0].ndim
                assert all(i.ndim == ndim for i in img)
                img = [np.expand_dims(img, -1) if ndim < 3 else i for i in img]
                img = [np.ascontiguousarray(i.transpose(2, 0, 1)) for i in img]
                stacked_image = np.stack(img, axis=0)
                img = stacked_image
            else:
                assert isinstance(img, np.ndarray), f"Expect img to be 'np.ndarray', but got {type(img)}"
                if img.ndim == 4:
                    img = np.ascontiguousarray(img.transpose(0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                elif img.ndim == 3:
                    img = np.ascontiguousarray(img.transpose(2, 0, 1)) # (H, W, C) -> (C, H, W)
                else:
                    raise RuntimeError
                
            if key in self.stack_keys:
                results[key] = DataContainer(to_tensor(img), stack=True)
            else:
                results[key] = DataContainer(to_tensor(img), stack=False)
        
        if 'depths' in results:
            depths = results['depths']
            depths = np.stack(depths, axis=0)
            results['depths'] = depths
        if 'cloud_list' in results:
            cloud_list = results['cloud_list']
            cloud_list = np.stack(cloud_list, axis=0)
            results['cloud_list'] = cloud_list
        if 'model_list' in results:
            model_list = results['model_list']
            model_list = np.stack(model_list, axis=0)
            results['model_list'] = model_list

        for key in results.get('annot_fields'):
            if key not in results or 'masks' in key:
                continue
            if key in self.stack_keys:
                results[key] = DataContainer(to_tensor(results[key]), stack=True)
            else:
                results[key] = DataContainer(to_tensor(results[key]), stack=False)
        
        if results.get('mask_fields', False):
            for field in results.get('mask_fields', ['masks']):
                if isinstance(results[field], (list, tuple)):
                    masks = results[field]
                    height, width = results[field][0].height, results[field][0].width
                    masks = BitmapMasks(masks, height, width)
                    results[field] = DataContainer(masks, cpu_only=True)
                else:
                    results[field] = DataContainer(results[field], cpu_only=True)
        return results

@PIPELINES.register_module()
class Collect:
    def __init__(self, 
                keys=('img',), 
                annot_keys=None,
                meta_keys=('img_path', 'ori_shape', 'ori_k', 'k',
                            'img_shape', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys
        self.annot_keys = annot_keys
        
    
    def __call__(self, results):
        data, annot, img_meta = {}, {}, {}
        if self.annot_keys is None:
            annot_keys = results.get('annot_fields')
        else:
            annot_keys = self.annot_keys
        for key in self.meta_keys:
            img_meta[key] = results[key]
        for key in annot_keys:
            if key not in annot_keys:
                continue
            annot[key] = results[key]
        for key in self.keys:
            data[key] = results[key]
        data['img_metas'] = DataContainer(img_meta, cpu_only=True)
        if len(annot) > 0:
            # when testing without annotation, don't register 'annots' key
            data['annots'] = annot
        return data


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

@PIPELINES.register_module()
class RepeatSample:
    def __init__(self, 
                repeat_times:int):
        self.repeat_times = repeat_times
    
    def __call__(self, results):
        results['img'] = self.repeat(results['img'])
        for k in results.get('annot_fields'):
            if k not in results:
                continue
            results[k] = self.repeat(results[k])
        if 'transform_matrix' in results:
            results['transform_matrix'] = self.repeat(results['transform_matrix'])
        return results


    def repeat(self, data, target_dims=None):
        if isinstance(data, np.ndarray):
            if target_dims is not None:
                if data.ndim ==  target_dims-1:
                    return np.repeat(data[None], self.repeat_times, axis=0)
                elif data.dim == target_dims:
                    return np.repeat(data, self.repeat_times, axis=0)
                else:
                    raise RuntimeError(f"Expect data to have {target_dims} or {target_dims-1} dimensions, but got {data.ndim}")
            else:
                return np.repeat(data, self.repeat_times, axis=0)
        elif isinstance(data, list):
            return [ele.copy() for ele in data for _ in range(self.repeat_times)]
            # new_data = []
            # for _ in range(self.repeat_times):
            #     new_data.extend([ele.copy() for ele in data])
            # return new_data
        elif isinstance(data, tuple):
            return tuple([ele.copy() for ele in data for _ in range(self.repeat_times)])
            # new_data = []
            # for _ in range(self.repeat_times):
            #     new_data.extend([ele.copy() for ele in data])
            # return tuple(new_data)
        else:
            raise RuntimeError(f"Not supported data type:{type(data)}")


@PIPELINES.register_module()
class MultiScaleAug:
    # TODO: Support Flip
    def __init__(self,
                transforms:list,
                scales:list=[0.9, 1., 1.1]) -> None:
        assert transforms[0]['type'] == 'Crop'
        pipelines = []
        for scale in scales:
            pipeline = deepcopy(transforms)
            pipeline[0]['size_range'] = (scale, scale)
            pipelines.append(Compose(pipeline))
        self.pipelines = pipelines


    def __call__(self, results):
        results_list = []
        for pipeline in self.pipelines:
            _results = deepcopy(results)
            results_list.append(pipeline(_results))
        
        # format aug data
        aug_data = {}
        new_keys = list(set(results_list[0].keys()) - set(results.keys()))
        new_keys += ['img', 'img_shape', 'k', 'ref_bboxes']
        for r in results_list:
            r['img'] = np.stack(r['img'])
        for k in results_list[0]:
            if k in new_keys:
                if isinstance(results_list[0][k], np.ndarray):
                    aug_data[k] = np.stack([r[k] for r in results_list])
                else:
                    aug_data[k] = [r[k] for r in results_list]
            else:
                aug_data[k] = results_list[0][k]
        return aug_data