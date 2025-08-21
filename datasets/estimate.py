import mmcv
import numpy as np
from os import path as osp
import itertools
from bop_toolkit_lib.inout import load_depth
from .builder import DATASETS
from .base_dataset import BaseDataset
from .supervise_refine import SuperviseTrainDataset



@DATASETS.register_module()
class SuperviseEstimationDataset(SuperviseTrainDataset):
    '''
    This class is used to train a 6D pose estimator which works on the image patch.
    
    Args:
        data_root (str): Image data root.
        image_list (str): Path to image list txt.
        pipeline (list[dict]): Processing pipelines.
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
                min_visib_fract=0, 
                min_visib_px_num=0, 
                sample_num=1, 
                anneal_visib_fract_rate=None, 
                label_mapping: dict = None, 
                target_label: list = None, 
                meshes_eval: str = None, 
                mesh_symmetry: dict = ..., 
                mesh_diameter: list = ...):
        super().__init__(
            data_root, image_list, pipeline, gt_annots_root, keypoints_json, 
            keypoints_num, class_names, min_visib_fract, min_visib_px_num, 
            sample_num, anneal_visib_fract_rate, 
            label_mapping, target_label, 
            meshes_eval, mesh_symmetry, mesh_diameter)

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
        gt_rotations, gt_translations, gt_labels, gt_bboxes, gt_keypoints_3d = [], [], [], [], []
        gt_mask_paths = []
        for i in range(gt_obj_num):
            ori_obj_id = gt_pose_annots[i]['obj_id']
            obj_id = ori_obj_id
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
            gt_keypoints_3d.append(self.keypoints_3d[ori_obj_id - 1])
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
        gt_keypoints_3d = np.stack(gt_keypoints_3d, axis=0)
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

        results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d')]
        results_dict['bbox_fields'] = ['gt_bboxes', 'ref_bboxes']
        results_dict['mask_fields'] = ['gt_masks']
        results_dict['label_fields'] = ['labels']
        
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields'] \
                                        + list(itertools.chain(*results_dict['pose_fields'])) + ['k', 'gt_keypoints_2d', 'gt_keypoints_3d_camera']
        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields']
        results_dict['gt_rotations'] = gt_rotations
        results_dict['gt_translations'] = gt_translations
        results_dict['gt_keypoints_3d'] = gt_keypoints_3d
        results_dict['labels'] = gt_labels
        results_dict['ori_gt_rotations'] = gt_rotations.copy()
        results_dict['ori_gt_translations'] = gt_translations.copy()
        results_dict['gt_bboxes'] = gt_bboxes
        results_dict['ref_bboxes'] = gt_bboxes.copy()
        results_dict['gt_mask_path'] = gt_mask_paths
        results_dict['k'] = results_dict['ori_k'] = k
        results_dict['img_path'] = img_path
        # data processing and augmentation pipeline
        results_dict = self.transformer(results_dict)

        return results_dict








@DATASETS.register_module()
class EstimationDataset(BaseDataset):
    '''
    This class is used to perfrom 6D pose estimation task.
    We require detected bboxes as input to crop the target region and resize it to a fixed resolution, 
    then the target pose is estimated from from the corp-and-resized patch.
    '''
    def __init__(self,
                data_root,
                image_list,
                pipeline,
                ref_bboxes_root,
                keypoints_json,
                keypoints_num,
                class_names : tuple,
                load_depth: bool=False,
                depth_ext: str='png',
                score_thr : float = 0.0,
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
            target_label=target_label,
            keypoints_num=keypoints_num,
            meshes_eval=meshes_eval,
            mesh_symmetry=mesh_symmetry,
            mesh_diameter=mesh_diameter
        )
        self.load_depth = load_depth
        self.score_thr = score_thr
        self.ref_bboxes_root = ref_bboxes_root
        self.info_json_tmpl = "{:06d}/scene_gt_info.json"
        self.camera_json_tmpl = osp.join(self.data_root, "{:06}/scene_camera.json")
        self.mask_path_tmpl = "{:06d}/mask_visib/{:06}_{:06}.png"
        self.ref_seq_annots = self._load_pose_annots()
        self.depth_ext = depth_ext
        self.depth_path_tmpl = "{}/{:06d}/depth/{:06d}.{}"
        
    
    def _load_pose_annots(self):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        ref_seq_pose_annots = dict()
        for sequence in sequences:
            ref_info_json_path = osp.join(self.ref_bboxes_root, self.info_json_tmpl.format(int(sequence)))
            camera_json_path = self.camera_json_tmpl.format(int(sequence))
            camera_annots = mmcv.load(camera_json_path)
            ref_seq_pose_annots[sequence] = dict(camera=camera_annots)
            ref_seq_pose_annots[sequence].update(ref_info=mmcv.load(ref_info_json_path))
        return ref_seq_pose_annots
    
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        depth_path = self.depth_path_tmpl.format(self.data_root, int(seq_name), img_id, self.depth_ext)
        ref_seq_annots = self.ref_seq_annots[seq_name]

        # load reference bboxes
        if str(img_id) in ref_seq_annots['ref_info']:
            ref_info = ref_seq_annots['ref_info'][str(img_id)]
        else:
            ref_info = ref_seq_annots['ref_info']["{:06}".format(img_id)]

        # load camera intrisic
        if str(img_id) in ref_seq_annots['camera']:
            camera_annots = ref_seq_annots['camera'][str(img_id)]
        else:
            camera_annots = ref_seq_annots['camera']["{:06}".format(img_id)]
        

        ref_obj_num = len(ref_info)
        ref_bboxes, ref_labels, ref_keypoints_3d = [], [], []
        for i in range(ref_obj_num):
            ori_obj_id = ref_info[i]['obj_id']
            obj_id = ori_obj_id
            if self.target_label is not None:
                if obj_id not in self.target_label:
                    continue
            if self.label_mapping is not None:
                if obj_id not in self.label_mapping:
                    continue
                obj_id = self.label_mapping[obj_id]
            score = ref_info[i]['score']
            if score <= self.score_thr:
                continue
            ref_keypoints_3d.append(self.keypoints_3d[ori_obj_id - 1])
            ref_bboxes.append(np.array(ref_info[i]['bbox_obj'], dtype=np.float32).reshape(-1))
            ref_labels.append(obj_id)
        
        if len(ref_bboxes) == 0:
            raise RuntimeError(f"No valid bboxes in {img_path}")
        
        ref_bboxes = np.stack(ref_bboxes, axis=0)
        # xywh --> xyxy
        ref_bboxes[:, 2:] = ref_bboxes[:, 2:] + ref_bboxes[:, :2]
        ref_labels = np.array(ref_labels, dtype=np.int64) - 1
        ref_keypoints_3d = np.stack(ref_keypoints_3d, axis=0)
        ref_obj_num = len(ref_bboxes)
        k_orig = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k_orig[None], repeats=ref_obj_num, axis=0)

        results_dict = dict()
        results_dict['bbox_fields'] = ['ref_bboxes']
        results_dict['label_fields'] = ['labels']
        results_dict['depth_fields'] = ['depths'] if self.load_depth else []
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['label_fields'] + results_dict['depth_fields'] \
                                        + ['k', 'ref_keypoints_3d', 'transform_matrix', 'ori_k']
        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['label_fields']
        results_dict['ref_keypoints_3d'] = ref_keypoints_3d
        results_dict['labels'] = ref_labels
        results_dict['ref_bboxes'] = ref_bboxes
        results_dict['k'] = k
        results_dict['ori_k'] = k_orig
        results_dict['img_path'] = img_path
        results_dict['ori_ref_bboxes'] = ref_bboxes.copy()
        if self.load_depth:
            results_dict['depths'] = load_depth(depth_path) * camera_annots['depth_scale']
        results_dict = self.transformer(results_dict)
        if results_dict is None:
            raise RuntimeError("Data pipeline is broken")
        return results_dict


@DATASETS.register_module()
class EstimationValDataset(SuperviseEstimationDataset):
    '''
    This class is used to perfrom 6D pose estimation task.
    We require detected bboxes as input to crop the target region and resize it to a fixed resolution, 
    then the target pose is estimated from from the corp-and-resized patch.
    '''
    def __init__(self, 
                data_root, 
                image_list, 
                pipeline, 
                gt_annots_root: str, 
                keypoints_json: str, 
                keypoints_num: int, 
                class_names: tuple, 
                label_mapping: dict = None, 
                target_label: list = None, 
                meshes_eval: str = None, 
                mesh_symmetry: dict = ..., 
                mesh_diameter: list = ...):
        super().__init__(
            data_root, image_list, pipeline=pipeline, gt_annots_root=gt_annots_root, 
            keypoints_json=keypoints_json, keypoints_num=keypoints_num, class_names=class_names, 
            label_mapping=label_mapping, target_label=target_label, 
            meshes_eval=meshes_eval, mesh_symmetry=mesh_symmetry, mesh_diameter=mesh_diameter)
    
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


        obj_num = len(gt_infos)
        ref_bboxes, ref_labels, ref_keypoints_3d = [], [], []
        gt_rotations, gt_translations = [], []
        for i in range(obj_num):
            ori_obj_id = gt_pose_annots[i]['obj_id']
            obj_id = ori_obj_id
            if self.label_mapping is not None:
                obj_id = self.label_mapping[obj_id]
            if self.target_label is not None:
                if obj_id not in self.target_label:
                    continue
            ref_keypoints_3d.append(self.keypoints_3d[ori_obj_id - 1])
            ref_bboxes.append(np.array(gt_infos[i]['bbox_obj'], dtype=np.float32).reshape(-1))
            ref_labels.append(obj_id)
            gt_rotations.append(np.array(gt_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_translations.append(np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))

        ref_bboxes = np.stack(ref_bboxes, axis=0)
        # xywh --> xyxy
        ref_bboxes[:, 2:] = ref_bboxes[:, 2:] + ref_bboxes[:, :2]
        ref_labels = np.array(ref_labels, dtype=np.int64) - 1
        ref_keypoints_3d = np.stack(ref_keypoints_3d, axis=0)
        ref_obj_num = len(ref_bboxes)
        k_orig = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k_orig[None], repeats=ref_obj_num, axis=0)
        gt_rotations = np.stack(gt_rotations, axis=0)
        gt_translations = np.stack(gt_translations, axis=0)


        results_dict = dict()
        results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d')]
        results_dict['bbox_fields'] = ['ref_bboxes']
        results_dict['label_fields'] = ['labels']
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['label_fields'] + \
                ['k', 'gt_keypoints_2d', 'ref_keypoints_3d', 'transform_matrix', 'ori_k']
        results_dict['gt_rotations'] = gt_rotations
        results_dict['gt_translations'] = gt_translations
        results_dict['gt_keypoints_3d'] = ref_keypoints_3d
        results_dict['ref_keypoints_3d'] = ref_keypoints_3d
        results_dict['labels'] = ref_labels
        results_dict['ref_bboxes'] = ref_bboxes
        results_dict['k'] = k
        results_dict['ori_k'] = k_orig
        results_dict['img_path'] = img_path

        results_dict = self.transformer(results_dict)
        return results_dict
