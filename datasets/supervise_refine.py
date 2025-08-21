import os
import itertools
import mmcv
import numpy as np
from os import path as osp
from pathlib import Path
from typing import Sequence, Optional
from .builder import DATASETS
from .base_dataset import BaseDataset
import multiprocessing as mp
from terminaltables import AsciiTable
from bop_toolkit_lib.inout import load_depth
from .bop_object_utils import load_objs, get_model_info


@DATASETS.register_module()
class SuperviseTrainDataset(BaseDataset):
    '''
    This class is used to perfrom 6D pose refinement task.
    Normally, for the refinement task, we need a reference pose, which can be produced by any pose estimation methods.
    
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
                data_root,
                image_list,
                pipeline,
                gt_annots_root: str,
                keypoints_json: str,
                keypoints_num: int,
                class_names: tuple,
                load_depth: bool=False,
                load_point_clouds: bool=False,
                depth_ext: str='png',
                min_visib_fract=0.,
                min_visib_px_num=0,
                sample_num=1,
                anneal_visib_fract_rate=None,
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
            keypoints_num=keypoints_num,
            meshes_eval=meshes_eval,
            mesh_symmetry=mesh_symmetry,
            mesh_diameter=mesh_diameter,
            target_label=target_label
        )
        self.load_depth = load_depth
        self.load_point_clouds = load_point_clouds
        self.min_visib_fract = min_visib_fract
        self.sample_num = sample_num
        self.gt_annots_root = gt_annots_root
        self.pose_json_tmpl = "{:06d}/scene_gt.json"
        self.info_json_tmpl = "{:06d}/scene_gt_info.json"
        self.camera_json_tmpl = osp.join(self.gt_annots_root, "{:06}/scene_camera.json")
        self.mask_path_tmpl = "{:06d}/mask_visib/{:06}_{:06}.png"
        self.gt_seq_pose_annots = self._load_pose_annots()
        self.min_visib_px_num = min_visib_px_num
        self.depth_ext = depth_ext
        self.depth_path_tmpl = "{}/{:06d}/depth/{:06d}.{}"
        obj, obj_ids = load_objs(model_path=os.path.join(data_root.rsplit('/', 1)[0], 'models'), sample_num=mesh_sample_num, n_template_view=0)
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.objects = obj
        self.obj_idxs = obj_idxs
        # set a multi processing shared number to record step
        # step = mp.Value('i', 0)
        # self.step = step
        # self.min_visib_fract_start = self.min_visib_fract
        # self.anneal_visib_fract_rate = anneal_visib_fract_rate
        # self.cal_total_sample_num()
    
    def cal_total_sample_num(self):
        table_data = [['class'] + list(self.class_names) + ['total']]
        sample_num_per_obj = {name:{'total_sample_num':0, 'valid_sample_num':0} for name in self.class_names}
        for sequence in self.gt_seq_pose_annots:
            gt_seq_infos, gt_seq_pose_annots = self.gt_seq_pose_annots[sequence]['gt_info'], self.gt_seq_pose_annots[sequence]['pose']
            for k in gt_seq_infos:
                gt_img_infos = gt_seq_infos[k]
                gt_img_pose_annots = gt_seq_pose_annots[k]
                for i in range(len(gt_img_infos)):
                    obj_info, obj_annot = gt_img_infos[i], gt_img_pose_annots[i]
                    ori_label = obj_annot['obj_id']
                    sample_num_per_obj[self.class_names[ori_label-1]]['total_sample_num'] += 1
                    if self.label_mapping is not None:
                        if ori_label not in self.label_mapping:
                            continue
                        label = self.label_mapping[ori_label]
                    else:
                        label = ori_label
                    if self.target_label is not None:
                        if label not in self.target_label:
                            continue
                    # if obj_info['visib_fract'] < self.min_visib_fract:
                    #     continue
                    sample_num_per_obj[self.class_names[ori_label-1]]['valid_sample_num'] += 1
    
        for k in ['total_sample_num', 'valid_sample_num']:
            table_data.append(
                [k] + [sample_num_per_obj[name][k] for name in sample_num_per_obj] + [sum([sample_num_per_obj[name][k] for name in sample_num_per_obj])]
            )
        self.total_sample_num = AsciiTable(table_data).table


    def _load_pose_annots(self):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        gt_seq_pose_annots = dict()
        for sequence in sequences:
            gt_pose_json_path = osp.join(self.gt_annots_root, self.pose_json_tmpl.format(int(sequence)))
            gt_info_json_path = osp.join(self.gt_annots_root, self.info_json_tmpl.format(int(sequence)))
            camera_json_path = self.camera_json_tmpl.format(int(sequence))
            gt_pose_annots = mmcv.load(gt_pose_json_path)
            gt_infos = mmcv.load(gt_info_json_path)
            camera_annots = mmcv.load(camera_json_path)
            gt_seq_pose_annots[sequence] = dict(pose=gt_pose_annots, camera=camera_annots, gt_info=gt_infos)
        return gt_seq_pose_annots
    
    def anneal_visib_thr(self):
        if self.anneal_visib_fract_rate is None:
            self.min_visib_fract = self.min_visib_fract_start
        else:
            self.min_visib_fract = self.min_visib_fract_start - self.anneal_visib_fract_rate * self.step.value


    
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        
        gt_seq_annots = self.gt_seq_pose_annots[seq_name]
        # load ground truth pose annots
        if str(img_id) in gt_seq_annots['pose']:
            gt_pose_annots = gt_seq_annots['pose'][str(img_id)]
        else:
            gt_pose_annots = gt_seq_annots['pose']["{:06}".format(img_id)]
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
            visib_fract = gt_infos[i]['visib_fract']
            if visib_fract < self.min_visib_fract:
                continue
            visib_px_count = gt_infos[i]['px_count_visib']
            if visib_px_count < self.min_visib_px_num:
                continue
            gt_rotations.append(np.array(gt_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_translations.append(np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))
            gt_labels.append(obj_id)
            gt_bboxes.append(np.array(gt_infos[i]['bbox_obj'], dtype=np.float32).reshape(-1))
            if 'mask_id' in gt_infos[i]:
                mask_path = osp.join(self.data_root, self.mask_path_tmpl.format(int(seq_name), img_id, gt_infos[i]['mask_id']))
            else:
                mask_path = osp.join(self.data_root, self.mask_path_tmpl.format(int(seq_name), img_id, i))
            gt_mask_paths.append(mask_path)
       
        if len(gt_labels) == 0:
            return None
        gt_rotations = np.stack(gt_rotations, axis=0)
        gt_translations = np.stack(gt_translations, axis=0)
        gt_labels = np.array(gt_labels, dtype=np.int64) - 1
        gt_keypoints_3d = self.keypoints_3d[gt_labels]
        gt_bboxes = np.stack(gt_bboxes, axis=0)
        # ground truth bboxes are xywh format
        gt_bboxes[..., 2:] = gt_bboxes[..., :2] + gt_bboxes[..., 2:]
        obj_num = len(gt_rotations)

        if self.sample_num == -1:
            sample_num = obj_num
        else:
            sample_num = self.sample_num
        choosen_obj_index = np.random.choice(list(range(obj_num)), sample_num)
        gt_rotations = gt_rotations[choosen_obj_index]
        gt_translations = gt_translations[choosen_obj_index]
        gt_labels = gt_labels[choosen_obj_index]
        gt_bboxes = gt_bboxes[choosen_obj_index]
        gt_keypoints_3d = gt_keypoints_3d[choosen_obj_index]
        gt_mask_paths = np.array(gt_mask_paths)[choosen_obj_index].tolist()
        
        k = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k[None], repeats=sample_num, axis=0)
        results_dict = dict()

        # results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d'), ('rotations', 'translations', 'keypoints_3d')]
        # results_dict['bbox_fields'] = ['gt_bboxes', 'bboxes']   #-# changed by qy 20240317
        results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d'), ('ref_rotations', 'ref_translations', 'ref_keypoints_3d')]
        results_dict['bbox_fields'] = ['gt_bboxes', 'ref_bboxes']
        results_dict['mask_fields'] = ['gt_masks']
        results_dict['label_fields'] = ['labels']

        #-# add depth 040406
        depth_path = self.depth_path_tmpl.format(self.data_root, int(seq_name), img_id, self.depth_ext)
        results_dict['depth_fields'] = ['depths'] if self.load_depth else []
        if self.load_depth:
            results_dict['depths'] = load_depth(depth_path) * camera_annots['depth_scale']
        
        #-# add models 240511
        results_dict['point_cloud_fields'] = ['model_list', 'model_diameter_list', 'cloud_list'] if self.load_point_clouds else []
        if self.load_point_clouds:
            model_list = [np.array(get_model_info(self.objects[label])[0]) for label in gt_labels]
            model_diameter_list = [self.objects[label].diameter for label in gt_labels]
            # print("img_path", img_path, 'label', gt_labels[0])
            results_dict['model_list'] = model_list
            results_dict['model_diameter_list'] = model_diameter_list

        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields'] \
                                        + list(itertools.chain(*results_dict['pose_fields'])) + ['k'] + results_dict['depth_fields'] \
                                        + results_dict['point_cloud_fields']

        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields']
        results_dict['gt_rotations'] = gt_rotations
        results_dict['gt_translations'] = gt_translations
        results_dict['gt_keypoints_3d'] = gt_keypoints_3d
        results_dict['ref_keypoints_3d'] = gt_keypoints_3d
        results_dict['ori_gt_rotations'] = gt_rotations.copy()
        results_dict['ori_gt_translations'] = gt_translations.copy()
        results_dict['labels'] = gt_labels
        results_dict['gt_bboxes'] = gt_bboxes
        results_dict['gt_mask_path'] = gt_mask_paths
        results_dict['k'] = results_dict['ori_k'] = k
        results_dict['img_path'] = img_path
        results_dict = self.transformer(results_dict)
        
        # if results_dict['annots']['depth_valid_flag'] is not None:
        #     if not results_dict['annots']['depth_valid_flag']:
        #         return None
        # elif results_dict['annots']['depth_valid_flag'] is None:
        #     return None
        # convert the multiprocessing cached number to
        # add 1 to the multiprocessing cached number of processed images
        # self.step.value += 1
        # self.anneal_visib_thr()

        return results_dict



@DATASETS.register_module()
class UnsuperviseTrainDataset(BaseDataset):
    '''
    This class is used to perfrom 6D pose refinement task.
    Normally, for the refinement task, we need a reference pose, which can be produced by any pose estimation methods.
    
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
                data_root: str,
                image_list: str,
                pipeline: Sequence[dict],
                ref_annots_root: str,
                keypoints_json: str,
                keypoints_num: int,
                class_names: tuple,
                load_depth: bool=False,
                depth_ext: str='png',
                filter_invalid_pose:bool=False,
                depth_range: Optional[tuple]=None,
                sample_num: int=1,
                label_mapping: dict = None,
                target_label: list = None,
                meshes_eval: str = None,
                mesh_symmetry: dict = {},
                mesh_diameter: list = []):
        super().__init__(
            data_root=data_root,
            image_list=image_list,
            keypoints_json=keypoints_json,
            pipeline=pipeline,
            class_names=class_names,
            label_mapping=label_mapping,
            keypoints_num=keypoints_num,
            meshes_eval=meshes_eval,
            mesh_symmetry=mesh_symmetry,
            mesh_diameter=mesh_diameter,
            target_label=target_label
        )
        self.load_depth = load_depth
        self.sample_num = sample_num
        self.ref_annots_root = ref_annots_root
        self.pose_json_tmpl = "{:06d}/scene_gt.json"
        self.info_json_tmpl = "{:06d}/scene_gt_info.json"
        self.camera_json_tmpl = osp.join(self.data_root, "{:06}/scene_camera.json")
        self.mask_path_tmpl = "{:06d}/mask_visib/{:06}_{:06}.png"
        self.seq_pose_annots = self._load_pose_annots()
        self.filter_invalid_pose = filter_invalid_pose
        self.depth_range = depth_range
        self.depth_ext = depth_ext
        self.depth_path_tmpl = "{}/{:06d}/depth/{:06d}.{}"
    


    def _load_pose_annots(self):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        ref_seq_pose_annots = dict()
        for sequence in sequences:
            ref_pose_json_path = osp.join(self.ref_annots_root, self.pose_json_tmpl.format(int(sequence)))
            camera_json_path = self.camera_json_tmpl.format(int(sequence))
            gt_pose_json_path = osp.join(self.data_root, self.pose_json_tmpl.format(int(sequence)))
            ref_pose_annots = mmcv.load(ref_pose_json_path)
            camera_annots = mmcv.load(camera_json_path)
            gt_pose_annots = mmcv.load(gt_pose_json_path)
            ref_seq_pose_annots[sequence] = dict(pose=ref_pose_annots, camera=camera_annots, gt_pose=gt_pose_annots)
        return ref_seq_pose_annots


    
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        depth_path = self.depth_path_tmpl.format(self.data_root, int(seq_name), img_id, self.depth_ext)
        seq_annots = self.seq_pose_annots[seq_name]
        # load reference pose annots
        if str(img_id) in seq_annots['pose']:
            ref_pose_annots = seq_annots['pose'][str(img_id)]
        else:
            ref_pose_annots = seq_annots['pose']["{:06}".format(img_id)]
        # load camera intrisic
        if str(img_id) in seq_annots['camera']:
            camera_annots = seq_annots['camera'][str(img_id)]
        else:
            camera_annots = seq_annots['camera']["{:06}".format(img_id)]
        # load ground truth pose annots
        if str(img_id) in seq_annots['gt_pose']:
            gt_pose_annots = seq_annots['gt_pose'][str(img_id)]
        else:
            gt_pose_annots = seq_annots['gt_psoe']["{:06}".format(img_id)]
        
        
        ref_obj_num = len(ref_pose_annots)
        ref_rotations, ref_translations, gt_rotations, gt_translations, ref_labels = [], [], [], [], []
        gt_obj_ids = [p['obj_id'] for p in gt_pose_annots]
        for i in range(ref_obj_num):
            obj_id = ref_pose_annots[i]['obj_id']
            matched_gt_index = [j for j, gt_obj_id in enumerate(gt_obj_ids) if gt_obj_id == obj_id]
            # TODO: remove this assumption
            if len(matched_gt_index) == 0:
                continue
            # we assume one obejct appear only once.
            matched_gt_index = matched_gt_index[0]
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
            ref_rotations.append(np.array(ref_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_rotations.append(np.array(gt_pose_annots[matched_gt_index]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_translations.append(np.array(gt_pose_annots[matched_gt_index]['cam_t_m2c'], dtype=np.float32).reshape(-1))
            ref_translations.append(translation)
            ref_labels.append(obj_id)
            


        if len(ref_labels) == 0:
            return None
        ref_rotations = np.stack(ref_rotations, axis=0)
        ref_translations = np.stack(ref_translations, axis=0)
        ref_labels = np.array(ref_labels, dtype=np.int64) - 1
        ref_keypoints_3d = self.keypoints_3d[ref_labels]
        gt_rotations = np.stack(gt_rotations, axis=0)
        gt_translations = np.stack(gt_translations, axis=0)

        
        obj_num = len(ref_rotations)
        if self.sample_num == -1:
            sample_num = obj_num
        else:
            sample_num = self.sample_num
        choosen_obj_index = np.random.choice(list(range(obj_num)), sample_num)
        ref_rotations = ref_rotations[choosen_obj_index]
        ref_translations = ref_translations[choosen_obj_index]
        ref_labels = ref_labels[choosen_obj_index]
        ref_keypoints_3d = ref_keypoints_3d[choosen_obj_index]
        gt_rotations = gt_rotations[choosen_obj_index]
        gt_translations = gt_translations[choosen_obj_index]
        
        k = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k[None], repeats=sample_num, axis=0)
        results_dict = dict()

        results_dict['pose_fields'] = [('rotations', 'translations', 'keypoints_3d'), ('valid_gt_rotations', 'valid_gt_translations', 'valid_gt_keypoints_3d')]
        results_dict['bbox_fields'] = ['bboxes']
        results_dict['mask_fields'] = []
        results_dict['label_fields'] = ['labels']
        #-# add depth -20240322
        results_dict['depth_fields'] = ['depths'] if self.load_depth else []
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields'] + results_dict['depth_fields'] \
                                        + list(itertools.chain(*results_dict['pose_fields'])) + ['k']
        if self.load_depth:
            results_dict['depths'] = load_depth(depth_path) * camera_annots['depth_scale']

        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields']
        results_dict['ref_keypoints_3d'] = ref_keypoints_3d
        results_dict['rotations'] = ref_rotations
        results_dict['translations'] = ref_translations
        results_dict['valid_gt_rotations'] = gt_rotations
        results_dict['valid_gt_translations'] = gt_translations
        results_dict['valid_gt_keypoints_3d'] = ref_keypoints_3d
        results_dict['labels'] = ref_labels
        results_dict['k'] = results_dict['ori_k'] = k
        results_dict['img_path'] = img_path
        results_dict = self.transformer(results_dict)
        return results_dict


@DATASETS.register_module()
class PseudoSuperviseTrainDataset(BaseDataset):
    '''
    This class is used to perfrom 6D pose refinement task.
    Normally, for the refinement task, we need a reference pose, which can be produced by any pose estimation methods.
    
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
                data_root,
                image_list,
                pipeline,
                gt_annots_root: str,
                keypoints_json: str,
                keypoints_num: int,
                class_names: tuple,
                sample_num=1,
                label_mapping: dict = None,
                target_label: list = None,
                meshes_eval: str = None,
                mesh_symmetry: dict = {},
                mesh_diameter: list = []):
        super().__init__(
            data_root=data_root,
            image_list=image_list,
            keypoints_json=keypoints_json,
            pipeline=pipeline,
            class_names=class_names,
            label_mapping=label_mapping,
            keypoints_num=keypoints_num,
            meshes_eval=meshes_eval,
            mesh_symmetry=mesh_symmetry,
            mesh_diameter=mesh_diameter,
            target_label=target_label
        )
        self.sample_num = sample_num
        self.gt_annots_root = gt_annots_root
        self.pose_json_tmpl = "{:06d}/scene_gt.json"
        self.info_json_tmpl = "{:06d}/scene_gt_info.json"
        self.camera_json_tmpl = osp.join(self.gt_annots_root, "{:06}/scene_camera.json")
        self.mask_path_tmpl = "{:06d}/mask_visib/{:06}_{:06}.png"
        self.gt_seq_pose_annots = self._load_pose_annots()



    def _load_pose_annots(self):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        gt_seq_pose_annots = dict()
        for sequence in sequences:
            gt_pose_json_path = osp.join(self.gt_annots_root, self.pose_json_tmpl.format(int(sequence)))
            camera_json_path = self.camera_json_tmpl.format(int(sequence))
            gt_pose_annots = mmcv.load(gt_pose_json_path)
            camera_annots = mmcv.load(camera_json_path)
            gt_seq_pose_annots[sequence] = dict(pose=gt_pose_annots, camera=camera_annots)
        return gt_seq_pose_annots
    
    def anneal_visib_thr(self):
        if self.anneal_visib_fract_rate is None:
            self.min_visib_fract = self.min_visib_fract_start
        else:
            self.min_visib_fract = self.min_visib_fract_start - self.anneal_visib_fract_rate * self.step.value


    
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        
        gt_seq_annots = self.gt_seq_pose_annots[seq_name]
        # load ground truth pose annots
        if str(img_id) in gt_seq_annots['pose']:
            gt_pose_annots = gt_seq_annots['pose'][str(img_id)]
        else:
            gt_pose_annots = gt_seq_annots['pose']["{:06}".format(img_id)]
        # load camera intrisic
        if str(img_id) in gt_seq_annots['camera']:
            camera_annots = gt_seq_annots['camera'][str(img_id)]
        else:
            camera_annots = gt_seq_annots['camera']["{:06}".format(img_id)]
        
        
        # we assume one obejct appear only once.
        gt_obj_num = len(gt_pose_annots)
        gt_rotations, gt_translations, gt_labels = [], [], []
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
            translation = np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1)
            if self.filter_invalid_pose:
                if translation[-1] > self.depth_range[-1] or translation[-1] < self.depth_range[0]:
                    continue
            gt_rotations.append(np.array(gt_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_translations.append(np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))
            gt_labels.append(obj_id)
            mask_path = osp.join(self.data_root, self.mask_path_tmpl.format(int(seq_name), img_id, i))
            gt_mask_paths.append(mask_path)
       
        if len(gt_labels) == 0:
            return None
        gt_rotations = np.stack(gt_rotations, axis=0)
        gt_translations = np.stack(gt_translations, axis=0)
        gt_labels = np.array(gt_labels, dtype=np.int64) - 1
        gt_keypoints_3d = self.keypoints_3d[gt_labels]
        obj_num = len(gt_rotations)

        if self.sample_num == -1:
            sample_num = obj_num
        else:
            sample_num = self.sample_num
        choosen_obj_index = np.random.choice(list(range(obj_num)), sample_num)
        gt_rotations = gt_rotations[choosen_obj_index]
        gt_translations = gt_translations[choosen_obj_index]
        gt_labels = gt_labels[choosen_obj_index]
        gt_keypoints_3d = gt_keypoints_3d[choosen_obj_index]
        gt_mask_paths = np.array(gt_mask_paths)[choosen_obj_index].tolist()
        
        k = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k[None], repeats=sample_num, axis=0)
        results_dict = dict()

        results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d'), ('rotations', 'translations', 'keypoints_3d')]
        results_dict['bbox_fields'] = ['bboxes']
        # results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d'), ('ref_rotations', 'ref_translations', 'ref_keypoints_3d')]
        # results_dict['bbox_fields'] = ['gt_bboxes', 'ref_bboxes']
        results_dict['mask_fields'] = ['gt_masks']
        results_dict['label_fields'] = ['labels']
        
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields'] \
                                        + list(itertools.chain(*results_dict['pose_fields'])) + ['k']
        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields']
        results_dict['gt_rotations'] = gt_rotations
        results_dict['gt_translations'] = gt_translations
        results_dict['gt_keypoints_3d'] = gt_keypoints_3d
        results_dict['ref_keypoints_3d'] = gt_keypoints_3d
        results_dict['ori_gt_rotations'] = gt_rotations.copy()
        results_dict['ori_gt_translations'] = gt_translations.copy()
        results_dict['labels'] = gt_labels
        results_dict['gt_mask_path'] = gt_mask_paths
        results_dict['k'] = results_dict['ori_k'] = k
        results_dict['img_path'] = img_path
        results_dict = self.transformer(results_dict)
        return results_dict