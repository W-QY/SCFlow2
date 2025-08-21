from typing import Optional, Sequence
import itertools
import mmcv
import numpy as np
import os
from os import path as osp
from pathlib import Path
from bop_toolkit_lib.inout import load_depth
from .builder import DATASETS
from .base_dataset import BaseDataset
from .bop_object_utils import load_objs, get_model_info






@DATASETS.register_module()
class RefineDataset(BaseDataset):
    def __init__(self,
                data_root:str,
                image_list:str,
                pipeline:Sequence[dict],
                ref_annots_root:str,
                keypoints_json:str,
                keypoints_num:int,
                gt_annots_root:str=None,
                filter_invalid_pose:bool=False,
                depth_range: Optional[tuple]=None,
                class_names : Optional[tuple]=None,
                load_depth: bool=False,
                load_mask: bool=False,
                load_point_clouds: bool=False,
                depth_ext: str='png',
                label_mapping: dict = None,
                target_label: list = None,
                meshes_eval: str = None,
                mesh_symmetry: dict = {},
                mesh_diameter: list = [],
                mesh_sample_num: int = 1024,
                ):
        super().__init__(
            data_root=data_root,
            image_list=image_list,
            keypoints_json=keypoints_json,
            pipeline=pipeline,
            class_names=class_names,
            label_mapping=label_mapping,
            target_label=target_label,
            keypoints_num=keypoints_num,
            meshes_eval=meshes_eval,
            mesh_symmetry=mesh_symmetry,
            mesh_diameter=mesh_diameter
        )
        self.ref_annots_root = ref_annots_root
        self.gt_annots_root = data_root if gt_annots_root is None else gt_annots_root
        self.pose_json_tmpl = "{:06d}/scene_gt.json"
        self.info_json_tmpl = "{:06d}/scene_gt_info.json"
        self.camera_json_tmpl = osp.join(self.gt_annots_root, "{:06}/scene_camera.json")
        self.mask_path_tmpl = "{:06d}/mask_visib/{:06}_{:06}.png"
        self.filter_invalid_pose = filter_invalid_pose
        self.depth_range = depth_range
        self.gt_seq_pose_annots, self.ref_seq_pose_annots = self._load_pose_annots()
        self.load_depth = load_depth
        self.load_mask = load_mask
        self.load_point_clouds = load_point_clouds
        self.depth_ext = depth_ext
        self.depth_path_tmpl = "{}/{:06d}/depth/{:06d}.{}"
        if self.load_point_clouds:
            obj, obj_ids = load_objs(model_path=meshes_eval, sample_num=mesh_sample_num, n_template_view=0)
            obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
            self.objects = obj
            self.obj_idxs = obj_idxs

    def _load_pose_annots(self):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        ref_seq_pose_annots = dict()
        gt_seq_pose_annots = dict()
        for sequence in sequences:
            ref_pose_json_path = osp.join(self.ref_annots_root, self.pose_json_tmpl.format(int(sequence)))
            gt_pose_json_path = osp.join(self.gt_annots_root, self.pose_json_tmpl.format(int(sequence)))
            ref_info_json_path = osp.join(self.ref_annots_root, self.info_json_tmpl.format(int(sequence)))
            gt_info_json_path = osp.join(self.gt_annots_root, self.info_json_tmpl.format(int(sequence)))
            camera_json_path = self.camera_json_tmpl.format(int(sequence))
            gt_pose_annots = mmcv.load(gt_pose_json_path)
            ref_pose_annots = mmcv.load(ref_pose_json_path)
            camera_annots = mmcv.load(camera_json_path)
            gt_infos = mmcv.load(gt_info_json_path)
            ref_seq_pose_annots[sequence] = dict(pose=ref_pose_annots, camera=camera_annots)
            gt_seq_pose_annots[sequence] = dict(pose=gt_pose_annots, camera=camera_annots, gt_info=gt_infos)
            if Path(ref_info_json_path).exists():
                ref_seq_pose_annots[sequence].update(ref_info=mmcv.load(ref_info_json_path))
        return gt_seq_pose_annots, ref_seq_pose_annots
    
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        gt_seq_annots, ref_seq_annots = self.gt_seq_pose_annots[seq_name], self.ref_seq_pose_annots[seq_name]
        depth_path = self.depth_path_tmpl.format(self.data_root, int(seq_name), img_id, self.depth_ext)

        # load ground truth pose annots
        if str(img_id) in gt_seq_annots['pose']:
            gt_pose_annots = gt_seq_annots['pose'][str(img_id)]
        else:
            gt_pose_annots = gt_seq_annots['pose']["{:06}".format(img_id)]
        
        # load referece pose annots     
        if str(img_id) in ref_seq_annots['pose']:
            ref_pose_annots = ref_seq_annots['pose'][str(img_id)]
        # else:
        #     ref_pose_annots = ref_seq_annots['pose']["{:06}".format(img_id)]
        else:   # debug
            print("This image has no valid coarse info: ", img_path)
            return None

        # load camera intrisic
        if str(img_id) in gt_seq_annots['camera']:
            camera_annots = gt_seq_annots['camera'][str(img_id)]
        else:
            camera_annots = gt_seq_annots['camera']["{:06}".format(img_id)]
        
        # load ground truth annotation related info, e.g., bbox, bbox_visib
        if str(img_id) in gt_seq_annots['gt_info']:
            gt_infos = gt_seq_annots['gt_info'][str(img_id)]
        else:
            gt_infos = gt_seq_annots['gt_info']["{:06}".format(img_id)]
        
        
        # we assume one obejct appear only once.
        gt_obj_num = len(gt_pose_annots)
        gt_rotations, gt_translations, gt_labels, gt_bboxes = [], [], [], []
        gt_mask_paths = []
        for i in range(gt_obj_num):
            obj_id = gt_pose_annots[i]['obj_id']
            if self.target_label is not None:
                if obj_id not in self.target_label:
                    continue
            if self.label_mapping is not None:
                if obj_id not in self.label_mapping:
                    continue
                obj_id = self.label_mapping[obj_id]
            px_count_visib = gt_infos[i]['px_count_visib']
            if px_count_visib == 0:
                continue
            gt_labels.append(obj_id)
            gt_rotations.append(np.array(gt_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_translations.append(np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))
            gt_bboxes.append(np.array(gt_infos[i]['bbox_obj'], dtype=np.float32).reshape(-1))
            mask_path = osp.join(self.gt_annots_root, self.mask_path_tmpl.format(int(seq_name), img_id, i))
            gt_mask_paths.append(mask_path)

        if len(gt_rotations) == 0:
            # raise RuntimeError(f"{img_path} found no gt")
            print("This image has no valid coarse info: ", img_path)
            return None
        gt_rotations = np.stack(gt_rotations, axis=0)
        gt_translations = np.stack(gt_translations, axis=0)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        gt_bboxes = np.stack(gt_bboxes, axis=0)
        # ground truth bboxes are xywj format
        gt_bboxes[..., 2:] = gt_bboxes[..., :2] + gt_bboxes[..., 2:]
        gt_obj_num = len(gt_rotations)
        
        formatted_gt_rotations, formatted_gt_translations, formatted_gt_bboxes, formatted_gt_mask_paths = [], [], [], []
        ref_obj_num = len(ref_pose_annots)
        if ref_obj_num > 0:
            ref_rotations, ref_translations, ref_labels = [], [], []
            for i in range(ref_obj_num):
                obj_id = ref_pose_annots[i]['obj_id']
                if self.target_label is not None:
                    if obj_id not in self.target_label:
                        continue
                if self.label_mapping is not None:
                    if obj_id not in self.label_mapping:
                        continue
                    obj_id = self.label_mapping[obj_id]
                translation = np.array(ref_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1)
                if self.filter_invalid_pose:
                    if translation[-1] > self.depth_range[-1] or translation[-1] < self.depth_range[0]:
                        continue
                if obj_id not in gt_labels:
                    continue
                ref_rotations.append(np.array(ref_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
                ref_translations.append(translation)
                ref_labels.append(obj_id)

                gt_index = np.nonzero(gt_labels == obj_id)[0][0]
                formatted_gt_rotations.append(gt_rotations[gt_index])
                formatted_gt_translations.append(gt_translations[gt_index])
                formatted_gt_bboxes.append(gt_bboxes[gt_index])
                formatted_gt_mask_paths.append(gt_mask_paths[gt_index])
        else:
            ref_rotations = np.zeros((0, 3, 3), dtype=np.float32)
            ref_translations = np.zeros((0, 3), dtype=np.float32)
            ref_keypoints_3d = np.zeros((0, 8, 3), dtype=np.float32)
        if len(ref_translations) == 0:
            print("This image has no valid coarse info: ", img_path)
            return None
        ref_translations = np.stack(ref_translations, axis=0)
        ref_rotations = np.stack(ref_rotations, axis=0)
        ref_labels = np.array(ref_labels, dtype=np.int64) - 1
        keypoints_3d = self.keypoints_3d[ref_labels]
        formatted_gt_rotations = np.stack(formatted_gt_rotations, axis=0)
        formatted_gt_translations = np.stack(formatted_gt_translations, axis=0)
        formatted_gt_bboxes = np.stack(formatted_gt_bboxes, axis=0)
        k_orig = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k_orig[None], repeats=ref_translations.shape[0], axis=0)
        

        results_dict = dict()
        results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d'), ('ref_rotations', 'ref_translations', 'ref_keypoints_3d')]
        results_dict['bbox_fields'] = ['gt_bboxes', 'ref_bboxes']
        results_dict['label_fields'] = ['labels']
        # results_dict['mask_fields'] = []  #-# add in 240512
        results_dict['mask_fields'] = ['gt_masks'] if self.load_mask else []
        #-# add depth -20240416
        results_dict['depth_fields'] = ['depths'] if self.load_depth else []
    
        if self.load_depth:
            results_dict['depths'] = load_depth(depth_path) * camera_annots['depth_scale']
        #-# add models 240512
        results_dict['point_cloud_fields'] = ['model_list', 'model_diameter_list', 'cloud_list'] if self.load_point_clouds else []
        if self.load_point_clouds:
            model_list = [np.array(get_model_info(self.objects[label])[0]) for label in ref_labels]
            model_diameter_list = [self.objects[label].diameter for label in ref_labels]
            results_dict['model_list'] = model_list
            results_dict['model_diameter_list'] = model_diameter_list

        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields'] + results_dict['depth_fields'] \
                                        + list(itertools.chain(*results_dict['pose_fields'])) + ['k', 'ori_k', 'transform_matrix'] + results_dict['point_cloud_fields']

        # results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['label_fields'] + results_dict['mask_fields']\
        #                                 + list(itertools.chain(*results_dict['pose_fields'])) + ['k', 'ori_k', 'transform_matrix']
        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['label_fields'] + results_dict['mask_fields']
        results_dict['ref_rotations'] = ref_rotations
        results_dict['ref_translations'] = ref_translations
        results_dict['gt_rotations'] = formatted_gt_rotations
        results_dict['gt_translations'] = formatted_gt_translations
        results_dict['ref_keypoints_3d'] = keypoints_3d
        results_dict['gt_keypoints_3d'] = keypoints_3d
        results_dict['keypoints_3d'] = keypoints_3d
        results_dict['labels'] = ref_labels
        results_dict['gt_bboxes'] = formatted_gt_bboxes
        results_dict['k'] = k
        results_dict['ori_k'] = k_orig
        results_dict['img_path'] = img_path
        results_dict['gt_mask_path'] = formatted_gt_mask_paths.copy()  # gt_mask_paths  add 240513
        results_dict['ori_gt_rotations'] = formatted_gt_rotations.copy()
        results_dict['ori_gt_translations'] = formatted_gt_translations.copy()
        results_dict['ori_ref_rotations'] = ref_rotations.copy()
        results_dict['ori_ref_translations'] = ref_translations.copy()
        results_dict = self.transformer(results_dict)
        if results_dict is None:
            raise RuntimeError(f"Data pipeline is broken for image {img_path}")

        return results_dict


@DATASETS.register_module()
class RefineTestDataset(BaseDataset):
    '''
    This class is used to perfrom 6D pose refinement task without pose matching. 
    Normally, for the refinement task, we need a reference pose, which can be produced by any pose estimation methods.
    If the bbox info exists, which may comes from an seperate detector, we load it.
    
    Args:
        data_root (str): Image data root.
        image_list (str): Path to image list txt.
        pipeline (list[dict]): Processing pipelines.
        ref_annots_root (str): Reference pose annotations root.
            Under this directory, the data should be formated as BOP formats. 
            But the mask annotations can be different.
            It should be organized as one of the following forms.
                1). Not exist. In this way, the reference mask will not be loaded.
                2). Json file where the mask is formated as PLE. In this way, the mask will be decoded.
                3). Png files. In this way, the mask will be used just as the BOP format.
        gt_annots_root (str | None): Ground truth annotations root, which should be specificed when evaluating the performance.
            Usually, it is the same as data_root.
        
    '''
    def __init__(self,
                data_root:str,
                image_list:str,
                pipeline:Sequence[dict],
                ref_annots_root:str,
                keypoints_json:str,
                keypoints_num:int,
                load_depth: bool=False,
                load_mask: bool=False,
                ref_mask_dir: str=None, 
                load_point_clouds: bool=False,
                crop_depth: bool=True,
                with_reference_bbox:bool=False,
                filter_invalid_pose:bool=False,
                depth_range: Optional[tuple]=None,
                class_names: Optional[Sequence]=None,
                label_mapping: Optional[dict]=None,
                target_label: Optional[Sequence]=None,
                meshes_eval: Optional[str]=None,
                depth_ext: str='png',
                mesh_symmetry: dict={},
                mesh_diameter: list=[],
                mesh_sample_num: int = 1024,
                ):
        super().__init__(
            data_root=data_root,
            image_list=image_list,
            keypoints_json=keypoints_json,
            pipeline=pipeline,
            class_names=class_names,
            label_mapping=label_mapping,
            target_label=target_label,
            keypoints_num=keypoints_num,
            meshes_eval=meshes_eval,
            mesh_symmetry=mesh_symmetry,
            mesh_diameter=mesh_diameter
        )
        self.with_reference_bbox = False
        self.load_depth = load_depth
        self.crop_depth = crop_depth
        self.load_mask = load_mask
        self.ref_mask_dir = ref_mask_dir
        self.load_point_clouds = load_point_clouds
        self.ref_annots_root = ref_annots_root
        self.pose_json_tmpl = "{:06d}/scene_gt.json"
        self.mask_json_tmpl = "{:06d}/scene_seg_info.json"
        self.info_json_tmpl = "{:06d}/scene_gt_info.json"
        self.camera_json_tmpl = osp.join(self.data_root, "{:06}/scene_camera.json")
        self.mask_path_tmpl = "{:06d}/mask_visib/{:06}_{:06}.png"
        self.depth_ext = depth_ext
        self.depth_path_tmpl = "{}/{:06d}/depth/{:06d}.{}"
        if "ipd" in self.data_root:
            self.camera_json_tmpl = osp.join(self.data_root, "{:06}/scene_camera_cam1.json")
            self.depth_path_tmpl = "{}/{:06d}/depth_cam1/{:06d}.{}"
            # self.pose_json_tmpl = "{:06d}/scene_gt_cam1.json"
        self.ref_seq_pose_annots = self._load_pose_annots()
        self.filter_invalid_pose = filter_invalid_pose
        self.depth_range = depth_range
        self.with_reference_bbox = with_reference_bbox
        obj, obj_ids = load_objs(model_path=meshes_eval, sample_num=mesh_sample_num, n_template_view=0)
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.objects = obj
        self.obj_idxs = obj_idxs
    
    def _load_pose_annots(self):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        ref_seq_pose_annots = dict()
        for sequence in sequences:
            ref_pose_json_path = osp.join(self.ref_annots_root, self.pose_json_tmpl.format(int(sequence)))
            camera_json_path = self.camera_json_tmpl.format(int(sequence))
            ref_pose_annots = mmcv.load(ref_pose_json_path)
            camera_annots = mmcv.load(camera_json_path)
            ref_seq_pose_annots[sequence] = dict(pose=ref_pose_annots, camera=camera_annots)
            if self.ref_mask_dir:
                ref_mask_json_path = osp.join(self.ref_mask_dir, self.mask_json_tmpl.format(int(sequence)))
                assert Path(ref_mask_json_path).exists()
                ref_seq_pose_annots[sequence].update(mask=mmcv.load(ref_mask_json_path))
            if self.with_reference_bbox:
                ref_info_json_path = osp.join(self.ref_annots_root, self.info_json_tmpl.format(int(sequence)))
                assert Path(ref_info_json_path).exists()
                ref_seq_pose_annots[sequence].update(info=mmcv.load(ref_info_json_path))
        return ref_seq_pose_annots
    
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        ref_seq_annots = self.ref_seq_pose_annots[seq_name]
        depth_path = self.depth_path_tmpl.format(self.data_root, int(seq_name), img_id, self.depth_ext)

        # load referece pose annots
        if self.ref_mask_dir:
            # load referece pose annots
            if (str(img_id) in ref_seq_annots['pose']) and (str(img_id) in ref_seq_annots['camera']) and (str(img_id) in ref_seq_annots['mask']):
                ref_pose_annots = ref_seq_annots['pose'][str(img_id)]
                camera_annots = ref_seq_annots['camera'][str(img_id)]
                mask_annots = ref_seq_annots['mask'][str(img_id)]
            else:
                print("This image has no valid coarse info: ", img_path)
                return None
        else:
            if str(img_id) in ref_seq_annots['pose']:
                ref_pose_annots = ref_seq_annots['pose'][str(img_id)]
            else:
                print("This image has no valid coarse info: ", img_path)
                return None
                # ref_pose_annots = ref_seq_annots['pose']["{:06}".format(img_id)]          # changed!! 1115
            # load camera intrisic
            if str(img_id) in ref_seq_annots['camera']:
                camera_annots = ref_seq_annots['camera'][str(img_id)]
            else:
                camera_annots = ref_seq_annots['camera']["{:06}".format(img_id)]

        # load reference bbox coming from a detector if exists
        if self.with_reference_bbox:
            try:
                ref_infos = ref_seq_annots['info'][str(img_id)]
            except:
                ref_infos = ref_seq_annots['info']["{:06}".format(img_id)]
        
        ref_obj_num = len(ref_pose_annots)
        assert ref_obj_num != 0, f"Image {img_path} has no references"
        ref_rotations, ref_translations, ref_labels, ref_bboxes, ref_mask_info, ref_mask_bbox = [], [], [], [], [], []
        gt_mask_paths = [osp.join(self.data_root, self.mask_path_tmpl.format(int(seq_name), img_id, 0))]
        for i in range(ref_obj_num):
            obj_id = ref_pose_annots[i]['obj_id']
            if self.target_label is not None:
                if obj_id not in self.target_label:
                    continue
            if self.label_mapping is not None:
                if obj_id not in self.label_mapping:
                    continue
                obj_id = self.label_mapping[obj_id]
            translation = ref_pose_annots[i]['cam_t_m2c']
            if self.filter_invalid_pose:
                if translation[-1] > self.depth_range[-1] or translation[-1] < self.depth_range[0]:
                    continue
            ref_rotations.append(np.array(ref_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            ref_translations.append(np.array(ref_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))
            ref_labels.append(obj_id)
            if self.ref_mask_dir:
                ref_mask_info.append(mask_annots[i])
                ref_mask_bbox.append(mask_annots[i]['bbox_obj'])
            if self.with_reference_bbox:
                ref_bboxes.append(np.array(ref_infos[i]['bbox_obj'], dtype=np.float32).reshape(4))

        if len(ref_rotations) == 0:
            # raise RuntimeError(f'No valid reference poses in {img_path}')
            print("This image has no valid coarse info: ", img_path)
            return None
        if self.ref_mask_dir:
            ref_mask_bbox = np.stack(ref_mask_bbox, axis=0)
            ref_mask_bbox[..., 2:] = ref_mask_bbox[..., :2] + ref_mask_bbox[..., 2:] # xywh to xyxy
        ref_rotations, ref_translations = np.stack(ref_rotations, axis=0), np.stack(ref_translations, axis=0)
        ref_labels = np.array(ref_labels, dtype=np.int64) - 1
        ref_keypoints_3d = self.keypoints_3d[ref_labels]
        if self.with_reference_bbox:
            ref_bboxes = np.stack(ref_bboxes, axis=0)
            ref_bboxes[..., 2:] = ref_bboxes[..., 2] + ref_bboxes[..., 2:] # xywh to xyxy
        ref_obj_num = len(ref_rotations)
        k_orig = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k_orig[None], repeats=ref_obj_num,  axis=0)

        results_dict = dict()
        results_dict['pose_fields'] = [('ref_rotations', 'ref_translations', 'ref_keypoints_3d')]
        results_dict['bbox_fields'] = ['ref_bboxes']
        results_dict['label_fields'] = ['labels']
        results_dict['mask_fields'] = []
        results_dict['depth_fields'] = ['depths'] if self.load_depth and self.crop_depth else []
        
        #-# add models 240830
        results_dict['mask_fields'] = ['gt_masks'] if self.load_mask else []
        results_dict['gt_mask_path'] = gt_mask_paths.copy()  # gt_mask_paths  add 240830
        if self.ref_mask_dir:
            results_dict['ref_mask_info'] = ref_mask_info  # gt_mask_paths  add 240830
            results_dict['ref_mask_bbox'] = ref_mask_bbox  # gt_mask_paths  add 240830
        if self.load_depth:
            results_dict['depths'] = load_depth(depth_path) * camera_annots['depth_scale']
        results_dict['point_cloud_fields'] = ['model_list', 'model_diameter_list', 'cloud_list'] if self.load_point_clouds else []
        if self.load_point_clouds:
            model_list = [np.array(get_model_info(self.objects[label])[0]) for label in ref_labels]
            model_diameter_list = [self.objects[label].diameter for label in ref_labels]
            results_dict['model_list'] = model_list
            results_dict['model_diameter_list'] = model_diameter_list
        #-------------------
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['label_fields'] \
                                        + results_dict['mask_fields'] + ['k', 'ori_k', 'transform_matrix'] \
                                        + list(itertools.chain(*results_dict['pose_fields'])) + results_dict['point_cloud_fields']
        results_dict['annot_fields'] += ['depths'] if self.load_depth else []
        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['label_fields']
        results_dict['ref_rotations'] = ref_rotations
        results_dict['ref_translations'] = ref_translations
        results_dict['ref_keypoints_3d'] = ref_keypoints_3d
        results_dict['keypoints_3d'] = ref_keypoints_3d
        results_dict['labels'] = ref_labels
        results_dict['k'] = k
        results_dict['ori_k'] = k_orig
        results_dict['img_path'] = img_path
        results_dict['ori_ref_rotations'] = ref_rotations.copy()
        results_dict['ori_ref_translations'] = ref_translations.copy()
        if self.with_reference_bbox:
            results_dict['ref_bboxes'] = ref_bboxes
        if self.load_depth:
            depth = load_depth(depth_path) * camera_annots['depth_scale']
            if not self.crop_depth:
                depth = [depth for _ in range(ref_obj_num)]
            results_dict['depths'] = depth
        results_dict = self.transformer(results_dict)
        if results_dict is None:
            raise RuntimeError(f"Data pipeline is broken for {img_path}, {ref_translations}")
        return results_dict