
from typing import Optional, Dict, Sequence
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .base_refiner import BaseRefiner
from ..utils import (
    solve_pose_by_pnp, get_flow_from_delta_pose_and_depth, 
    cal_epe, filter_flow_by_mask, get_2d_3d_corr_by_fw_flow,
    get_3d_3d_corr_by_fw_flow, solve_pose_by_ransac_kabsch)




class BaseFlowRefiner(BaseRefiner):
    def __init__(self, 
                encoder: Optional[Dict]=None, 
                decoder: Optional[Dict]=None, 
                seperate_encoder: bool=False, 
                filter_invalid_flow_by_mask: bool=False,
                filter_invalid_flow_by_depth: bool=False,
                renderer: Optional[Dict]=None, 
                render_augmentations: Optional[Sequence[Dict]]=None, 
                train_cfg: dict={}, 
                test_cfg: dict={}, 
                init_cfg: dict={}, 
                max_flow: int=400):
        super().__init__(encoder, decoder, seperate_encoder, renderer, render_augmentations, train_cfg, test_cfg, init_cfg, max_flow)
        self.filter_invalid_flow_by_mask = filter_invalid_flow_by_mask
        self.filter_invalid_flow_by_depth = filter_invalid_flow_by_depth
        self.solve_pose_space = test_cfg.get('solve_pose_space', 'transformed')
        assert self.solve_pose_space in ['origin', 'transformed'] 
    
    def get_flow(self):
        raise NotImplementedError

        
    def format_data_train_sup(self, data_batch):
        data = super().format_data_train_sup(data_batch)
        if self.filter_invalid_flow_by_depth:
            gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
            internel_k, labels = data['internel_k'], data['labels']
            render_outputs = self.renderer(gt_rotations, gt_translations, internel_k, labels)
            gt_rendered_depths = render_outputs['fragments'].zbuf
            gt_rendered_depths = gt_rendered_depths[..., 0]
            data.update(gt_rendered_depths = gt_rendered_depths)
            return data
        else:
            return data
        
    def random_sample_points(self, points_2d, points_3d, sample_points_num):
        assert len(points_2d) == len(points_3d)
        num_points = len(points_2d)
        if sample_points_num > num_points:
            return points_2d, points_3d
        rand_index = torch.randperm(num_points-1, device=points_2d.device)[:sample_points_num]
        return points_2d[rand_index], points_3d[rand_index]

    def topk_sample_points(self, points_2d, points_3d, confidence, sample_points_num):
        assert len(points_2d) == len(points_3d)
        num_points = len(points_2d)
        if sample_points_num > num_points:
            return points_2d, points_3d
        _, index = torch.topk(confidence, k=sample_points_num)
        return points_2d[index], points_3d[index]
    
    def sample_points(self, points_2d, points_3d, sample_cfg, points_confidence=None):
        sample_points_num = sample_cfg.get('num', 1000)
        sample_points_mode = sample_cfg.get('mode', 'random')
        if sample_points_mode == 'random':
            return self.random_sample_points(points_2d, points_3d, sample_points_num)
        else:
            return self.topk_sample_points(points_2d, points_3d, points_confidence, sample_points_num)
        
   

    def val_step(self, data_batch):
        pred_flow, data = self.forward(data_batch)
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        internel_k, rendered_depths, rendered_masks = data['internel_k'], data['rendered_depths'], data['rendered_masks']
        gt_masks = data['gt_masks']
        epe, epe_noc = self.eval_epe(
            pred_flow, rendered_depths, ref_rotations, ref_translations, internel_k, gt_rotations, gt_translations, rendered_masks, gt_masks, reduction='total_mean')
        log_vars = {'epe':epe.item(), 'epe_noc':epe_noc.item()}
        return dict(
            log_vars=log_vars
        )
    
    def eval_epe(self, batch_flow, rendered_depths, ref_rotations, ref_translations, internel_k, gt_rotations, gt_translations, gt_masks=None, reduction='mean'):
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        epe = cal_epe(gt_flow, batch_flow, reducion=reduction, max_flow=self.max_flow)
        if gt_masks is not None:
            noc_gt_flow = filter_flow_by_mask(gt_flow, gt_masks, self.max_flow)
            valid_mask = torch.sum(noc_gt_flow**2, dim=1).sqrt() < self.max_flow
            epe_noc = cal_epe(noc_gt_flow, batch_flow, valid_mask, reduction)
        else:
            epe_noc = epe
        return epe, epe_noc
    
    def gen_multiview(self, ref_rotations, ref_translations):
        # numpy version for multiview rotation generation, may slow down the speed
        rotation_perturb_cfg = self.test_cfg.get('rotation_perturb', {})
        angles = Rotation.from_matrix(ref_rotations.cpu().data.numpy()).as_euler('xyz', degrees=True)
        angle_x, angle_y, angle_z = angles[..., 0], angles[..., 1], angles[..., 2]
        angles = [angle_x, angle_y, angle_z]
        perturbed_poses = [(ref_rotations, ref_translations)]
        axis_order = {'x':0, 'y':1, 'z':2}
        for axis in rotation_perturb_cfg:
            assert axis in axis_order
            order = axis_order[axis]
            orig_angle = angles[order]
            angle_perturbs = rotation_perturb_cfg[axis]
            for angle_perturb in angle_perturbs:
                perturbed_angles = angles.copy()
                perturbed_angles[order] = orig_angle + angle_perturb
                perturbed_angles = np.stack(perturbed_angles, axis=-1)
                perturbed_rotations = Rotation.from_euler('xyz', perturbed_angles, degrees=True).as_matrix()
                perturbed_poses.append((
                    torch.from_numpy(perturbed_rotations).to(ref_rotations.device).to(torch.float32),
                    ref_translations
                ))
        
        translation_perturb_cfg = self.test_cfg.get('translation_perturb', {})
        x, y, z = ref_translations[..., 0], ref_translations[..., 1], ref_translations[..., 2]
        translations = [x, y, z]
        for axis in translation_perturb_cfg:
            assert axis in axis_order
            order = axis_order[axis]
            orig_trans = translations[order]
            offset_perturbs = translation_perturb_cfg[axis]
            for offset_perturb in offset_perturbs:
                perturbed_trans = translations.copy()
                perturbed_trans[order] = orig_trans + offset_perturb
                perturbed_poses.append(
                    (ref_rotations, torch.stack(perturbed_trans, dim=-1))
                )
        num_images = len(ref_rotations)
        perturbed_rotations_list, perturbed_translations_list = [], []
        for i in range(num_images):
            perturbed_rotations_list.append(
                torch.stack([r[i] for r, t in perturbed_poses], dim=0)
            )
            perturbed_translations_list.append(
                torch.stack([t[i] for r, t in perturbed_poses], dim=0)
            )
        return perturbed_rotations_list, perturbed_translations_list
    

    
    def format_multiview_test_data(self, data_batch):
        real_images, annots, meta_infos = data_batch['img'], data_batch['annots'], data_batch['img_metas']
        device = real_images[0].device

        ref_rotations, ref_translations = annots['ref_rotations'], annots['ref_translations']
        # gt_rotations, gt_translaions = annots['gt_rotations'], annots['gt_translations']
        labels, internel_k = annots['labels'], annots['k']
        ori_k, transform_matrixs = annots['ori_k'], annots['transform_matrix']

        per_img_patch_num = [len(images) for images in real_images]
        real_images = torch.cat(real_images)
        ref_rotations, ref_translations = torch.cat(ref_rotations, dim=0), torch.cat(ref_translations, dim=0)
        # gt_rotations, gt_translaions = torch.cat(gt_rotations, dim=0), torch.cat(gt_translaions, dim=0)
        labels = torch.cat(labels)
        internel_k = torch.cat(internel_k)

        transform_matrixs = torch.cat(transform_matrixs)
        ori_k = torch.cat([k[None].expand(patch_num, 3, 3) for k, patch_num in zip(ori_k, per_img_patch_num)])

        ref_rotations_list, ref_translations_list = self.gen_multiview(ref_rotations, ref_translations)
        rendered_images_list, rendered_depths_list, rendered_masks_list = [], [], []
        for i, (ref_rotations_per_sample, ref_translations_per_sample) in enumerate(zip(ref_rotations_list, ref_translations_list)):
            labels_per_sample = labels[i][None].expand(len(ref_rotations_per_sample))
            internel_k_per_sample = internel_k[i][None].expand(len(ref_rotations_per_sample), -1, -1)
            render_outputs = self.renderer(ref_rotations_per_sample, ref_translations_per_sample, internel_k_per_sample, labels_per_sample)
            rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
            rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
            rendered_depths = rendered_fragments.zbuf
            rendered_depths = rendered_depths[..., 0]
            rendered_masks = (rendered_depths > 0).to(torch.float32)
            rendered_images_list.append(rendered_images)
            rendered_depths_list.append(rendered_depths)
            rendered_masks_list.append(rendered_masks)

        output =  dict(
            real_images = real_images,
            render_images = rendered_images_list,
            render_depths = rendered_depths_list,
            render_masks = rendered_masks_list,
            ref_rotations = ref_rotations_list,
            ref_translations = ref_translations_list,
            internel_k = internel_k,
            ori_k = ori_k,
            labels = labels,
            transform_matrix = transform_matrixs,
            per_img_patch_num = per_img_patch_num
        )

        if 'depths' in annots:
            real_depths = torch.cat(annots['depths'], dim=0)
            output.update(real_depths=real_depths)
        return output

    def solve_pose(self, 
                batch_flow : torch.Tensor, 
                rendered_depths : torch.Tensor, 
                ref_rotations : torch.Tensor, 
                ref_translations : torch.Tensor, 
                internel_k : torch.Tensor, 
                labels : torch.Tensor, 
                per_img_patch_num : torch.Tensor, 
                occlusion: Optional[torch.Tensor]=None):
        batch_rotations, batch_translations = [], []
        num_images = len(rendered_depths)
        if occlusion is not None:
            occlusion_thresh = self.test_cfg.get('occ_thresh', 0.5)
            valid_mask = occlusion > occlusion_thresh
        else:
            valid_mask = None 
        points_corr = get_2d_3d_corr_by_fw_flow(batch_flow, rendered_depths, ref_rotations, ref_translations, internel_k, valid_mask)
        sample_points_cfg = self.test_cfg.get('sample_points', None) 
        retval_flag = []
        for i in range(num_images):
            ref_points_2d, tgt_points_2d, points_3d = points_corr[i]
            if sample_points_cfg is not None:
                if occlusion is not None:
                    points_confidence = occlusion[i, ref_points_2d[:, 1].to(torch.int64), ref_points_2d[:, 0].to(torch.int64)]
                    tgt_points_2d, points_3d = self.sample_points(tgt_points_2d, points_3d, sample_points_cfg, points_confidence)
                else:
                    tgt_points_2d, points_3d = self.sample_points(tgt_points_2d, points_3d, sample_points_cfg)

            rotation_pred, translation_pred, retval = solve_pose_by_pnp(tgt_points_2d, points_3d, internel_k[i], **self.test_cfg)
            if retval:
                rotation_pred = torch.from_numpy(rotation_pred)[None].to(torch.float32).to(ref_rotations.device)
                translation_pred = torch.from_numpy(translation_pred)[None].to(torch.float32).to(ref_rotations.device)
                retval_flag.append(True)
            else:
                rotation_pred = ref_rotations[i][None]
                translation_pred = ref_translations[i][None]
                retval_flag.append(False)
            batch_rotations.append(rotation_pred)
            batch_translations.append(translation_pred)
        
        batch_rotations = torch.split(torch.cat(batch_rotations), per_img_patch_num)
        batch_translations = torch.split(torch.cat(batch_translations), per_img_patch_num)
        batch_labels = torch.split(labels, per_img_patch_num)
        batch_scores = torch.split(torch.ones_like(labels, dtype=torch.float32), per_img_patch_num)
        batch_retval_flag = torch.split(torch.tensor(retval_flag, device=labels.device, dtype=torch.bool), per_img_patch_num)
        
        batch_rotations = [p[r] for p,r in zip(batch_rotations, batch_retval_flag)]
        batch_translations = [p[r] for p,r in zip(batch_translations, batch_retval_flag)]
        batch_labels = [l[r] for l,r in zip(batch_labels, batch_retval_flag)]
        batch_scores = [s[r] for s,r in zip(batch_scores, batch_retval_flag)]
        return dict(
            rotations=batch_rotations,
            translations=batch_translations,
            scores=batch_scores,
            labels=batch_labels,
        ) 
    
    def solve_pose_depth(self, 
                        batch_flow:torch.Tensor, 
                        rendered_depths:torch.Tensor, 
                        real_depths:torch.Tensor, 
                        ref_rotations:torch.Tensor, 
                        ref_translations:torch.Tensor, 
                        internel_k:torch.Tensor,
                        labels:torch.Tensor,
                        per_img_patch_num:torch.Tensor,
                        occlusion:Optional[torch.Tensor]=None):
        batch_rotations, batch_translations = [], []
        num_images = len(rendered_depths)
        valid_mask = None
        if occlusion is not None:
            occlusion_thresh = self.test_cfg.get('occ_thresh', 0.5)
            valid_mask = occlusion > occlusion_thresh
        
        points_corr = get_3d_3d_corr_by_fw_flow(
            batch_flow, rendered_depths, real_depths, ref_rotations, ref_translations, internel_k, valid_mask)
        sample_points_cfg = self.test_cfg.get('sample_points', None)

        for i in range(num_images):
            points_3d_camera_frame, points_3d_object_frame = points_corr[i]
            if sample_points_cfg is not None:
                points_3d_camera_frame, points_3d_object_frame = self.sample_points(points_3d_camera_frame, points_3d_object_frame, sample_points_cfg)
            rotation_pred, translation_pred, retval = solve_pose_by_ransac_kabsch(points_3d_camera_frame, points_3d_object_frame)
            if retval:
                rotation_pred, translation_pred = rotation_pred[None], translation_pred[None]
            else:
                rotation_pred, translation_pred = ref_rotations[i][None], ref_translations[i][None]
            batch_rotations.append(rotation_pred)
            batch_translations.append(translation_pred)
        batch_rotations = torch.split(torch.cat(batch_rotations), per_img_patch_num)
        batch_translations = torch.split(torch.cat(batch_translations), per_img_patch_num)
        batch_labels = torch.split(labels, per_img_patch_num)
        batch_scores = torch.split(torch.ones_like(labels, dtype=torch.float32), per_img_patch_num)
        return dict(
            rotations=batch_rotations,
            translations=batch_translations,
            scores=batch_scores,
            labels=batch_labels,
        )