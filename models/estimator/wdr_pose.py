from typing import Sequence, Optional
import torch
from torch.nn import functional as F
import mmcv
from mmcv.runner import BaseModule
import numpy as np
from .builder import ESTIMATORS
from models import build_backbone, build_decoder, build_loss, build_head
from models.utils import (
    MultiLevelAssigner, AnchorGenerator, TargetCoder, 
    solve_pose_by_ransac_kabsch,
    solve_pose_by_pnp, lift_2d_to_3d,
    remap_pose_to_origin_resoluaion,
    remap_points_to_origin_resolution)
from PIL import Image, ImageColor, ImageDraw
from matplotlib import pyplot as plt

def trans_paste(w, h, color, bg_img, alpha=1.0, box=(0, 0)):
    alpha = int(255*alpha)
    color = ImageColor.getrgb(color)
    color = tuple(list(color) + [alpha])
    fg_img = Image.new("RGBA", (h, w), color)
    bg_img.paste(fg_img, box, fg_img)
    return bg_img

def draw_bbox_text(drawobj, xmin, ymin, xmax, ymax, color, text=None, bd=2):
    drawobj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawobj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawobj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawobj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    if text:
        drawobj.text((xmin+3, ymin), text, fill='Red')

def draw_keypoint(drawobj:ImageDraw.ImageDraw, x, y, color, size=4, text=None):
    drawobj.ellipse((x-size, y-size, x+size, y+size), fill=color)
    if text is not None:
        drawobj.text((x, y-2*size), text, fill='Red')





@ESTIMATORS.register_module()
class WDRPose(BaseModule):
    def __init__(self,
                backbone: dict,
                neck: dict,
                head: dict,
                num_classes: int, 
                loss_cls: dict=None,
                use_depth: bool=False,
                ignore_not_valid_keypoints=True,
                loss_keypoint_3d: dict=None,
                loss_keypoint_2d: dict=None,
                keypoint_num: int=8,
                assigner : dict=dict(num_pos=10, pos_lambda=1., anchor_sizes=(128, 160, 192, 224, 256)),
                coder: dict=dict(normalizer=1/8, clip_border=True),
                anchor_generator: dict=dict(),
                vis_tensorboard: bool=False,
                train_cfg:dict=None,
                test_cfg:dict=dict(),
                init_cfg=None):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.keypoint_num = keypoint_num
        self.backbone = build_backbone(backbone)
        self.neck = build_decoder(neck)
        self.head = build_head(head)
        self.use_depth = use_depth
        self.ignore_not_valid_keypoints = ignore_not_valid_keypoints
        self.keypoint_3d_loss_func = None
        self.keypoint_2d_loss_func = None 
        if loss_keypoint_3d is not None:
            loss_keypoint_3d.update(type='ObjectSpaceLoss')
            self.keypoint_3d_loss_func = build_loss(loss_keypoint_3d)
        if loss_keypoint_2d is not None:
            loss_keypoint_2d.update(type='ImageSpaceLoss')
            self.keypoint_2d_loss_func = build_loss(loss_keypoint_2d)
        if loss_cls is not None:
            self.cls_loss_func = build_loss(loss_cls)
        self.assigner = MultiLevelAssigner(**assigner)
        self.anchor_generator = AnchorGenerator(**anchor_generator)
        self.coder = TargetCoder(**coder)
        self.train_cfg = train_cfg 
        self.test_cfg = test_cfg
        self.solve_pose_space = test_cfg.get('solve_pose_space', 'origin')
        assert self.solve_pose_space in ['origin', 'transformed']
        self.vis_tensorboard = vis_tensorboard


    def extract_feat(self, images:torch.Tensor, labels:Optional[torch.Tensor]):
        feats = self.backbone(images)
        feats = self.neck(feats)
        cls_scores, keypoint_preds = self.head(feats, labels)
        return cls_scores, keypoint_preds


    def train_step(self, data_batch, optimizer, **kwargs):
        data = self.format_train_data(data_batch)
        images, labels = data['images'], data['gt_labels']
        cls_scores, keypoint_preds = self.extract_feat(images, labels)
        outputs = self.loss(cls_scores, keypoint_preds, data)
        if self.vis_tensorboard:
            loss, log_vars, log_imgs = outputs
            return dict(
                loss = loss,
                log_vars = log_vars,
                log_imgs = log_imgs,
                num_samples = len(data_batch['img_metas']),
            )
        else:
            loss, log_vars = outputs
            return dict(
                loss = loss,
                log_vars = log_vars,
                num_samples = len(data_batch['img_metas']),
            )
    
    def format_train_data(self, data_batch):
        real_images, annots, meta_infos = data_batch['img'], data_batch['annots'], data_batch['img_metas']
        gt_rotations, gt_translations = annots['gt_rotations'], annots['gt_translations']
        gt_masks = [mask.to_tensor(dtype=torch.bool, device=gt_rotations[0].device) for mask in annots['gt_masks']]
        labels, internel_k, gt_bboxes = annots['labels'], annots['k'], annots['gt_bboxes']
        keypoints_3d, keypoints_3d_camera, keypoints_2d = annots['gt_keypoints_3d'], annots['gt_keypoints_3d_camera'], annots['gt_keypoints_2d']

        real_images = torch.cat(real_images)
        gt_rotations, gt_translations = torch.cat(gt_rotations, axis=0), torch.cat(gt_translations, axis=0)
        gt_masks = torch.cat(gt_masks, axis=0)
        gt_bboxes = torch.cat(gt_bboxes, axis=0)
        labels, internel_k = torch.cat(labels), torch.cat(internel_k)
        keypoints_3d, keypoints_3d_camera, keypoints_2d = torch.cat(keypoints_3d), torch.cat(keypoints_3d_camera), torch.cat(keypoints_2d)
        return dict(
            images=real_images,
            gt_rotations = gt_rotations,
            gt_translations = gt_translations,
            gt_masks = gt_masks,
            gt_bboxes = gt_bboxes,
            gt_labels = labels,
            internel_k = internel_k,
            keypoints_3d = keypoints_3d,
            keypoints_3d_camera = keypoints_3d_camera,
            keypoints_2d = keypoints_2d,
            meta_infos = meta_infos
        )
    
    def loss(self, cls_scores:Sequence[torch.Tensor], keypoint_preds:Sequence[torch.Tensor], data:dict):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_priors(featmap_sizes, device)
        mlvl_points = [(anchor[:, 0:2] + anchor[:, 2:])/2 for anchor in mlvl_anchors]
        image_h, image_w, _ = data['meta_infos'][0]['img_shape'][0]

        # each image only contains one annotation, including bbox, R, T, and label
        gt_bboxes, gt_labels, gt_masks = data['gt_bboxes'], data['gt_labels'], data['gt_masks']
        assigned_gt_inds_list, points_weight_list = [], []
        for gt_bbox, gt_mask in zip(gt_bboxes, gt_masks):
            assigned_gt_inds, points_weight = self.assigner.assign(mlvl_points, gt_bbox[None], gt_mask[None])
            assigned_gt_inds_list.append(assigned_gt_inds)
            points_weight_list.append(points_weight)
    
        internel_k, gt_3d_keypoints, gt_2d_keypoints  = data['internel_k'], data['keypoints_3d_camera'], data['keypoints_2d']

        # debug
        if False:
            self.debug(data['images'], gt_bboxes, gt_labels, gt_2d_keypoints, assigned_gt_inds_list, torch.cat(mlvl_anchors), data['meta_infos'])

        num_points = len(torch.cat(mlvl_points))
        internel_k_list = [k[None].expand(num_points, 3, 3) for k in internel_k]
        keypoints_2d_targets_list, keypoints_3d_targets_list, label_targets_list = [], [], []
        num_pos = 0
        for assigned_gt_inds, gt_2d_keypoint, gt_3d_keypoint, gt_label in zip(assigned_gt_inds_list, gt_2d_keypoints, gt_3d_keypoints, gt_labels):
            pos_inds = assigned_gt_inds > 0
            keypoints_2d_targets = gt_2d_keypoints.new_zeros((num_points, self.keypoint_num, 2))
            keypoints_3d_targets_camera_frame = gt_3d_keypoint.new_zeros((num_points, self.keypoint_num, 3))
            keypoints_2d_targets[pos_inds] = gt_2d_keypoint
            keypoints_3d_targets_camera_frame[pos_inds] = gt_3d_keypoint
            # background's label target is set to 1
            label_targets = gt_label.new_full((num_points, ), self.num_classes)
            label_targets[pos_inds] = gt_label
            keypoints_2d_targets_list.append(keypoints_2d_targets)
            keypoints_3d_targets_list.append(keypoints_3d_targets_camera_frame)
            label_targets_list.append(label_targets)
            num_pos += pos_inds.sum(0)

        num_images = len(gt_bboxes)
        pos_inds = torch.cat(assigned_gt_inds_list) > 0
        keypoints_2d_targets = torch.cat(keypoints_2d_targets_list, dim=0)
        keypoints_3d_targets = torch.cat(keypoints_3d_targets_list, dim=0)
        label_targets = torch.cat(label_targets_list, dim=0)
        points_weight = torch.cat(points_weight_list, dim=0)
        concat_anchors = torch.cat([torch.cat(mlvl_anchors, dim=0) for _ in range(num_images)])
        concat_internel_k = torch.cat(internel_k_list)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_images, -1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_keypoint_preds = [
            # (N, keypoint_num, 2, H, W) --> (N, H, W, keypoint_num, 2) -> (N, H*W, kepoint_num, 2)
            keypoint_pred.permute(0, 3, 4, 1, 2).reshape(num_images, -1, self.keypoint_num, 2)
            for keypoint_pred in keypoint_preds
        ]

        
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).reshape(-1, self.num_classes)
        flatten_keypoint_preds = torch.cat(flatten_keypoint_preds, dim=1).reshape(-1, self.keypoint_num, 2) 

        log_vars = {}
        loss_cls = self.cls_loss_func(flatten_cls_scores, label_targets, weight=points_weight, avg_factor=num_pos+num_images)
        log_vars.update(loss_cls=loss_cls.item())
        loss = 0
        loss += loss_cls
        if num_pos > 0:
            pos_anchors = concat_anchors[pos_inds]
            pos_internel_k = concat_internel_k[pos_inds]
            label_targets_pos = label_targets[pos_inds]
            pos_weights = points_weight[pos_inds]
            keypoints_2d_targets_pos = keypoints_2d_targets[pos_inds]
            keypoints_3d_targets_pos = keypoints_3d_targets[pos_inds]
            keypoints_2d_pred_pos = flatten_keypoint_preds[pos_inds]

            if self.ignore_not_valid_keypoints:
                valid_flag_x = (keypoints_2d_targets_pos[..., 0] >=0) & (keypoints_2d_targets_pos[..., 0] < image_w)
                valid_flag_y = (keypoints_2d_targets_pos[..., 1] >=0) & (keypoints_2d_targets_pos[..., 1] < image_h)
                valid_flag = (valid_flag_x & valid_flag_y)[..., None]
            else:
                valid_flag = torch.ones((num_pos, self.keypoint_num, 1), dtype=torch.bool, device=pos_anchors.device)
            
            pos_weights = pos_weights[:, None, None] * valid_flag
            if self.keypoint_3d_loss_func is not None:
                keypoint_3d_loss = self.keypoint_3d_loss_func(
                    keypoints_2d_pred_pos, 
                    target_3d=keypoints_3d_targets_pos,
                    label=label_targets_pos,
                    internel_k=pos_internel_k,
                    coder=self.coder,
                    anchors=pos_anchors,
                    avg_factor=num_pos+num_images,
                    weight=pos_weights,
                )
                log_vars.update(loss_keypoint_3d=keypoint_3d_loss.item())
                loss += keypoint_3d_loss
                
            if self.keypoint_2d_loss_func is not None:
                keypoint_2d_loss = self.keypoint_2d_loss_func(
                    keypoints_2d_pred_pos,
                    target_2d=keypoints_2d_targets_pos,
                    coder=self.coder,
                    anchors=pos_anchors,
                    avg_factor=num_pos+num_images,
                    weight=pos_weights,
                )
                log_vars.update(loss_keypoint_2d=keypoint_2d_loss.item())
                loss += keypoint_2d_loss
    
        log_vars.update(loss=loss.item())

        if self.vis_tensorboard:
            # visualize the first sample in a batch
            image_show_keypoint, image_show_assign, input_image = self.visualize(flatten_keypoint_preds[:num_points], gt_2d_keypoints[0], concat_anchors[:num_points], assigned_gt_inds_list[0], data['images'][0], data['meta_infos'][0])
            log_imgs = dict(
                keypoint_image = image_show_keypoint,
                assisgn_image = image_show_assign,
                input_image = input_image
            )
            return loss, log_vars, log_imgs
        else:
            return loss, log_vars

    def format_test_data(self, data_batch:dict):
        real_images, annots, meta_infos = data_batch['img'], data_batch['annots'], data_batch['img_metas']
        # comes from detector
        labels, internel_k, keypoints_3d = annots['labels'], annots['k'], annots['ref_keypoints_3d']
        ori_k, transform_matrixs = annots['ori_k'], annots['transform_matrix']

        per_img_patch_num = [len(images) for images in real_images]
        real_images = torch.cat(real_images)
        labels, internel_k, keypoints_3d = torch.cat(labels), torch.cat(internel_k), torch.cat(keypoints_3d)
        transform_matrixs = torch.cat(transform_matrixs)
        ori_k = torch.cat([k[None].expand(patch_num, 3, 3) for k, patch_num in zip(ori_k, per_img_patch_num)])
        image_shape_list = []
        for img_meta in meta_infos:
            image_shape_list.extend(img_meta['img_shape'])
        
        output =  dict(
            images = real_images,
            labels = labels,
            internel_k = internel_k,
            ori_k = ori_k,
            transform_matrix = transform_matrixs,
            keypoints_3d = keypoints_3d,
            per_img_patch_num = per_img_patch_num,
            meta_infos = meta_infos,
            image_shapes =image_shape_list
        )
        if 'gt_keypoints_2d' in annots:
            gt_keypoints_2d = torch.cat(annots['gt_keypoints_2d'])
            output.update(gt_keypoints_2d = gt_keypoints_2d)
        if 'depths' in annots:
            depths = torch.cat(annots['depths'], dim=0)
            output.update(depths=depths)
        return output


    def forward(self, data_batch:dict, return_loss=False):
        data = self.format_test_data(data_batch)
        images, labels = data['images'], data['labels']
        cls_scores, keypoint_preds = self.extract_feat(images, labels)
        if self.use_depth:
            return self.get_pose_with_depth(cls_scores, keypoint_preds, labels, data)
        else:
            return self.get_pose(cls_scores, keypoint_preds, labels, data)


    def get_pose_with_depth(self, cls_scores:Sequence[torch.Tensor], keypoint_preds:Sequence[torch.Tensor], labels:torch.Tensor, data:dict):
        '''Forward with depth, using Ransac-kabsch to solve the pose'''
        keypoints_2d = self.get_keypoints_2d(cls_scores, keypoint_preds, labels, data)
        device = cls_scores[0].device
        labels, keypoints_3d, internel_k = data['labels'], data['keypoints_3d'], data['internel_k']
        ori_k, transform_matrixs = data['ori_k'], data['transform_matrix']
        per_image_patch_num = data['per_img_patch_num']
        depths = data['depths']

        num_images = len(per_image_patch_num)
        patch_to_image_index = torch.cat([torch.full((patch_num, ), i, dtype=torch.int64) for i, patch_num in enumerate(per_image_patch_num)])
        batch_rotations, batch_translations = [], []
        valid_predcition_flag = torch.ones_like(patch_to_image_index, dtype=torch.bool)

        for i, fusion_keypoints_2d in enumerate(keypoints_2d):
            keypoints_3d_object_frame = keypoints_3d[i][None].expand(len(fusion_keypoints_2d), self.keypoint_num, 3).reshape(-1, 3)
            fusion_keypoints_2d = fusion_keypoints_2d.reshape(-1, 2)
            depth = depths[i]
            h, w = depth.shape
            normalized_keypoints_2d_x = fusion_keypoints_2d[:, 0] * 2 / w - 1. 
            normalized_keypoints_2d_y = fusion_keypoints_2d[:, 1] * 2 / h - 1.
            # shape (1, 1, keypoints_num, 2) --> (N, H_out, W_out, 2)
            normalized_keypoints = torch.stack([normalized_keypoints_2d_x, normalized_keypoints_2d_y], dim=-1)[None, None]
            # shape (1, 1, 1, keypoints_num) --> (N, C, H_out, W_out)
            keypoints_depth = F.grid_sample(
                depth[None, None],
                normalized_keypoints, 
                mode='bilinear',
                padding_mode='zeros',
            )
            keypoints_depth = keypoints_depth[0, 0, 0]
            keypoints_3d_camera_frame = lift_2d_to_3d(fusion_keypoints_2d[:, 0], fusion_keypoints_2d[:, 1], keypoints_depth, internel_k[i])
            rot, trans, retval = solve_pose_by_ransac_kabsch(keypoints_3d_camera_frame, keypoints_3d_object_frame, **self.test_cfg)
            valid_predcition_flag[i] = retval
            if retval:
                batch_rotations.append(rot)
                batch_translations.append(trans)
            else:
                batch_rotations.append(torch.zeros((3, 3)).to(device))
                batch_translations.append(torch.zeros((3,)).to(device))
        batch_rotations = list(torch.split(torch.stack(batch_rotations, dim=0), per_image_patch_num))
        batch_translations = list(torch.split(torch.stack(batch_translations, dim=0), per_image_patch_num))
        batch_labels = list(torch.split(labels, per_image_patch_num))
        batch_scores = list(torch.split(torch.ones_like(labels, dtype=torch.float32), per_image_patch_num))
        batch_valid_flag = list(torch.split(valid_predcition_flag, per_image_patch_num))
        batch_internel_k = list(torch.split(internel_k, per_image_patch_num))
        for i in range(num_images):
            batch_rotations[i] = batch_rotations[i][batch_valid_flag[i]]
            batch_translations[i] = batch_translations[i][batch_valid_flag[i]]
            batch_labels[i] = batch_labels[i][batch_valid_flag[i]]
            batch_scores[i] = batch_scores[i][batch_valid_flag[i]]
        if self.solve_pose_space != 'origin':
            batch_rotations, batch_translations = remap_pose_to_origin_resoluaion(batch_rotations, batch_translations, batch_internel_k, data['meta_infos'])
        return dict(
            rotations=batch_rotations,
            translations=batch_translations,
            scores=batch_scores,
            labels=batch_labels,
        )
                


    def get_pose(self, cls_scores:Sequence[torch.Tensor], keypoint_preds:Sequence[torch.Tensor], labels:torch.Tensor, data:dict):
        keypoints_2d = self.get_keypoints_2d(cls_scores, keypoint_preds, labels, data)
        labels, keypoints_3d, internel_k = data['labels'], data['keypoints_3d'], data['internel_k']
        ori_k, transform_matrixs = data['ori_k'], data['transform_matrix']
        per_image_patch_num = data['per_img_patch_num']
        num_images = len(per_image_patch_num)
        patch_to_image_index = torch.cat([torch.full((patch_num, ), i, dtype=torch.int64) for i, patch_num in enumerate(per_image_patch_num)])
        batch_rotations, batch_translations = [], []
        valid_predcition_flag = torch.ones_like(patch_to_image_index, dtype=torch.bool)

        for i, fusion_keypoints_2d in enumerate(keypoints_2d):
            keypoints_3d_per_img = keypoints_3d[i][None].expand(len(fusion_keypoints_2d), self.keypoint_num, 3).reshape(-1, 3)
            fusion_keypoints_2d = fusion_keypoints_2d.reshape(-1, 2)
            if self.solve_pose_space == 'transformed':
                internel_k_per_img = internel_k[i]
                rot, trans, retval = solve_pose_by_pnp(fusion_keypoints_2d, keypoints_3d_per_img, internel_k_per_img, **self.test_cfg)
            else:
                ori_k_per_img, transform_matrix_per_img = ori_k[i], transform_matrixs[i]
                mapped_fusion_keypoints_2d = remap_points_to_origin_resolution(fusion_keypoints_2d, transform_matrix_per_img)
                rot, trans, retval = solve_pose_by_pnp(mapped_fusion_keypoints_2d, keypoints_3d_per_img, ori_k_per_img, **self.test_cfg)
            valid_predcition_flag[i] = retval
            if retval:
                batch_rotations.append(rot)
                batch_translations.append(trans)
            else:
                batch_rotations.append(np.zeros((3, 3)))
                batch_translations.append(np.zeros(3, ))
        batch_rotations = list(torch.split(torch.from_numpy(np.stack(batch_rotations, axis=0)), per_image_patch_num))
        batch_translations = list(torch.split(torch.from_numpy(np.stack(batch_translations, axis=0)), per_image_patch_num))
        batch_labels = list(torch.split(labels, per_image_patch_num))
        batch_scores = list(torch.split(torch.ones_like(labels, dtype=torch.float32), per_image_patch_num))
        batch_valid_flag = list(torch.split(valid_predcition_flag, per_image_patch_num))
        batch_internel_k = list(torch.split(internel_k, per_image_patch_num))
        for i in range(num_images):
            batch_rotations[i] = batch_rotations[i][batch_valid_flag[i]]
            batch_translations[i] = batch_translations[i][batch_valid_flag[i]]
            batch_labels[i] = batch_labels[i][batch_valid_flag[i]]
            batch_scores[i] = batch_scores[i][batch_valid_flag[i]]
        if self.solve_pose_space != 'origin':
            batch_rotations, batch_translations = remap_pose_to_origin_resoluaion(batch_rotations, batch_translations, batch_internel_k, data['meta_infos'])
        return dict(
            rotations=batch_rotations,
            translations=batch_translations,
            scores=batch_scores,
            labels=batch_labels,
        )
    
    def get_keypoints_2d(self, cls_scores:Sequence[torch.Tensor], keypoint_preds:Sequence[torch.Tensor], labels:torch.Tensor, data:dict):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_priors(featmap_sizes, device)
        num_images = sum(data['per_img_patch_num'])    
        image_shapes = data['image_shapes']
        mlvl_scores, mlvl_keypoints, mlvl_anchors_select = [], [], []
        post_process_pre = self.test_cfg.get('post_process_pre', 1000)
        for cls_score_per_lvl, keypoint_pred_per_lvl, anchor_per_lvl in zip(cls_scores, keypoint_preds, mlvl_anchors):
            cls_score_per_lvl = cls_score_per_lvl.permute(0, 2, 3, 1).reshape(num_images, -1, self.num_classes).sigmoid()
            # select cls score accroding to the reference labels
            cls_score_per_lvl = cls_score_per_lvl[torch.arange(num_images), :, labels]
            keypoint_pred_per_lvl = keypoint_pred_per_lvl.permute(0, 3, 4, 1, 2).reshape(num_images, -1, self.keypoint_num, 2)
            anchor_per_lvl = anchor_per_lvl.expand(num_images, -1, -1)
            post_process_pre_per_lvl = post_process_pre if post_process_pre < keypoint_pred_per_lvl.size(1) else -1
            if post_process_pre_per_lvl > 0:
                _, topk_inds = torch.topk(cls_score_per_lvl, k=post_process_pre_per_lvl)
                batch_inds = torch.arange(num_images).view(-1, 1).expand_as(topk_inds).long()
                anchor_per_lvl = anchor_per_lvl[batch_inds, topk_inds, :]
                keypoint_pred_per_lvl = keypoint_pred_per_lvl[batch_inds, topk_inds, :]
                cls_score_per_lvl = cls_score_per_lvl[batch_inds, topk_inds]
            
            keypoint_pred_per_lvl_list = []
            for i in range(num_images):
                keypoint_pred_per_lvl_list.append(self.coder.decode(anchor_per_lvl[i], keypoint_pred_per_lvl[i], image_shapes[i]))
            keypoint_pred_per_lvl = torch.stack(keypoint_pred_per_lvl_list)
            mlvl_scores.append(cls_score_per_lvl)
            mlvl_keypoints.append(keypoint_pred_per_lvl)
            mlvl_anchors_select.append(anchor_per_lvl)
        
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_keypoints = torch.cat(mlvl_keypoints, dim=1)
        num_per_level = [s.size(1) for s in mlvl_scores]

        topk, score_thr, positive_lambda, anchor_sizes = list(map(self.test_cfg.get, ['topk', 'score_thr', 'positive_lambda', 'anchor_sizes']))
        predicted_keypoints_2d = []
        for i, (concat_mlvl_scores, concat_mlvl_keypoints) in enumerate(zip(batch_mlvl_scores, batch_mlvl_keypoints)):
            fusion_scores, fusion_keypoints, select_inds = self.multi_level_fusion(
                concat_mlvl_scores, concat_mlvl_keypoints, num_per_level, topk, score_thr, positive_lambda, anchor_sizes)
            if False:
                self.debug_pred_keypoints(fusion_keypoints, fusion_scores, mlvl_anchors[0][select_inds], data['images'][i], data['meta_infos'][0])
            predicted_keypoints_2d.append(fusion_keypoints)

        return predicted_keypoints_2d



    def multi_level_fusion(self,
                        multi_level_scores:torch.Tensor,
                        multi_level_keypoints:torch.Tensor,
                        num_per_level:Sequence[int],
                        topk:int,
                        score_thr:float,
                        positive_lambda:float,
                        anchor_sizes: Sequence[int]):
        '''
        Single class version, simplify logic compared to multi class version
        Args:
            multi_level_scores (torch.Tensor): shape (N)
            multi_level_keypoints (torch.Tensor): shape (N, 8, 2)
            num_per_level (list|tuple): num of predictions for each level, 
                sum(num_per_level)=N
            topk (int): select topk most confident predictions from multi level
            score_thr (float): the predictions with score below score_thr will be treated as valid predicions
            positive_lambda (float): determine the selection number for each level

        '''
        num_levels = len(num_per_level)
        valid_mask = multi_level_scores > score_thr 
        valid_scores = multi_level_scores[valid_mask]
        valid_keypoints = multi_level_keypoints[valid_mask]

        if valid_keypoints.numel() == 0:
            keypoint_num = multi_level_keypoints.size(1)
            return valid_scores.new_zeros((0)), valid_keypoints.new_zeros((0, keypoint_num, 2)), valid_mask
        most_confident_index = torch.argmax(valid_scores)
        most_confident_keypoint_pred = valid_keypoints[most_confident_index]
        bbox_size = max(most_confident_keypoint_pred[:, 0].max() - most_confident_keypoint_pred[:, 0].min(), 
                        most_confident_keypoint_pred[:, 1].max() - most_confident_keypoint_pred[:, 1].min())


        dk = torch.log2(bbox_size.view(1) / bbox_size.new_tensor(anchor_sizes).view(-1))
        nk = torch.exp(-positive_lambda * (dk * dk))
        nk = topk * nk / nk.sum()
        nk = (nk + 0.5).int()
        if topk ==  -1:
            nk[...] = -1

        multi_level_scores_list = torch.split(multi_level_scores, num_per_level, dim=0)
        multi_level_keypoints_list = torch.split(multi_level_keypoints, num_per_level, dim=0)
        multi_level_mask_list = torch.split(valid_mask, num_per_level, dim=0)
        
        scores_list, keypoints_list = [], []
        choosen_flag_list = []
        for i in range(num_levels):
            scores_level_i = multi_level_scores_list[i]
            keypoints_level_i = multi_level_keypoints_list[i]
            valid_mask_level_i = multi_level_mask_list[i]
            valid_pred_num = sum(valid_mask_level_i)
            choosen_flag = torch.zeros_like(valid_mask_level_i, dtype=torch.bool)
            
            if nk[i] == -1:
                choosen_num = valid_pred_num
            else:
                choosen_num = min(nk[i], valid_pred_num)
            if choosen_num > 0:
                scores_level_i, choosen_inds = torch.topk(scores_level_i, k=choosen_num)
                keypoints_list.append(keypoints_level_i[choosen_inds])
                scores_list.append(scores_level_i)
                choosen_flag[choosen_inds] = True
            choosen_flag_list.append(choosen_flag)
        choosen_flag = torch.cat(choosen_flag_list)
        if len(keypoints_list) == 0:
            keypoints = torch.zeros((0, 8, 2), device=multi_level_keypoints.device)
            scores = torch.zeros((0, 8, 2), device=multi_level_keypoints.device)
        else:
            keypoints = torch.cat(keypoints_list, dim=0)
            scores = torch.cat(scores_list, dim=0)
        return scores, keypoints, choosen_flag
    


    def visualize(self, pred_keypoints_2d, gt_keypoints_2d, anchors, assigned_gt_inds, image, image_meta):
        ''' Visualize for a single image
        '''
        image_np = mmcv.tensor2imgs(image[None], image_meta['img_norm_cfg']['mean'], image_meta['img_norm_cfg']['std'], to_rgb=False)[0]
        image_pil_show_keypoints = Image.fromarray(image_np.copy())
        image_pil_show_assign = Image.fromarray(image_np.copy())
        drawobj = ImageDraw.ImageDraw(image_pil_show_keypoints)
        # show ground truth keypoints
        for j in range(self.keypoint_num):
            draw_keypoint(drawobj, gt_keypoints_2d[j, 0], gt_keypoints_2d[j, 1], color='red', text=str(j))
        
        pos_flag = assigned_gt_inds > 0
        pos_anchors, pos_pred_keypoints_2d = anchors[pos_flag], pred_keypoints_2d[pos_flag]
        pos_pred_keypoints_2d = self.coder.decode(pos_anchors, pos_pred_keypoints_2d)
        w, h = (pos_anchors[:, 2] - pos_anchors[:, 0]) / self.anchor_generator.octave_base_scale, (pos_anchors[:, 3] - pos_anchors[:, 1]) / self.anchor_generator.octave_base_scale
        w, h = w.int(), h.int()
        cx, cy = (pos_anchors[:, 2] + pos_anchors[:, 0]) / 2, (pos_anchors[:, 3] + pos_anchors[:, 1]) / 2
        x1, y1 = (cx - w / 2).int(), (cy - h / 2).int()
        pos_anchor_num = pos_anchors.shape[0]
        for i in range(pos_anchor_num):
            # show assigned positive anchors(points)
            trans_paste(w[i]-2, h[i]-2, color='green', bg_img=image_pil_show_assign, alpha=0.8, box=(x1[i]-1, y1[i]-1))
            for j in range(self.keypoint_num):
                # show predicted keypoints by assigned positive sample
                draw_keypoint(drawobj, pos_pred_keypoints_2d[i, j, 0], pos_pred_keypoints_2d[i, j, 1], size=2, color='blue')
        return np.array(image_pil_show_keypoints), np.array(image_pil_show_assign), image_np
        


    def debug_pred_keypoints(self, fusion_keypoints, fusion_scores, anchors, image, image_meta, gt_keypoints=None):
        assert len(fusion_keypoints) == len(anchors) == len(fusion_scores)
        image_np = mmcv.tensor2imgs(image[None], image_meta['img_norm_cfg']['mean'], image_meta['img_norm_cfg']['std'], to_rgb=False)[0]
        pred_num = anchors.shape[0]
        image_pil_show_keypoints = Image.fromarray(image_np.copy())
        image_pil_show_cell = Image.fromarray(image_np.copy())
        drawobj = ImageDraw.ImageDraw(image_pil_show_keypoints)
        for i in range(pred_num):
            for j in range(self.keypoint_num):
                draw_keypoint(drawobj, fusion_keypoints[i, j, 0], fusion_keypoints[i, j, 1], size=2, color='red')
        if gt_keypoints is not None:
            for j in range(self.keypoint_num):
                draw_keypoint(drawobj, gt_keypoints[j, 0], gt_keypoints[j, 1], size=3, color='red')
        plt.imshow(image_pil_show_keypoints)
        plt.savefig('debug/pred_keypoints.png')
        
        w, h = (anchors[:, 2] - anchors[:, 0]) / self.anchor_generator.octave_base_scale, (anchors[:, 3] - anchors[:, 1]) / self.anchor_generator.octave_base_scale
        w, h = w.int(), h.int()
        cx, cy = (anchors[:, 2] + anchors[:, 0]) / 2, (anchors[:, 3] + anchors[:, 1]) / 2
        x1, y1 = (cx - w / 2).int(), (cy - h / 2).int()
        for i in range(pred_num):
            trans_paste(w[i]-2, h[i]-2, color='green', bg_img=image_pil_show_cell, alpha=fusion_scores[i], box=(x1[i]-1, y1[i]-1))
        plt.imshow(image_pil_show_cell)
        plt.savefig('debug/activate_cell.png')


    
    
    def debug_label_assign(self, images:torch.Tensor, gt_bboxes:torch.Tensor, gt_labels:torch.Tensor, gt_keypoints_2d:torch.Tensor, assigned_gt_inds:Sequence[torch.Tensor], multi_level_anchors:Sequence[torch.Tensor], img_metas:Sequence[dict]):
        images_np = mmcv.tensor2imgs(images, img_metas[0]['img_norm_cfg']['mean'], img_metas[0]['img_norm_cfg']['std'], to_rgb=False)
        for i, img_meta in enumerate(img_metas):
            assigned_gt_inds_per_img = assigned_gt_inds[i]
            image = Image.fromarray(images_np[i])
            resized_h, resized_w, _ = img_meta['img_shape'][0]
            image = image.resize((resized_w, resized_h))
            gt_bbox, gt_label = gt_bboxes[i], gt_labels[i]
            drawobj = ImageDraw.ImageDraw(image)
            draw_bbox_text(drawobj, gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], color='red')
            for j in range(self.keypoint_num):
                draw_keypoint(drawobj, gt_keypoints_2d[i, j, 0], gt_keypoints_2d[i, j, 1], color='red', text=str(j))
            

            # pos_anchor_box_index = torch.nonzero((assigned_gt_inds_per_img > 0), as_tuple=False).view(-1)
            # pos_anchor_box = multi_level_anchors[pos_anchor_box_index]
            # w, h = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) / self.anchor_generator.octave_base_scale, (pos_anchor_box[:, 3] - pos_anchor_box[:, 1]) / self.anchor_generator.octave_base_scale
            # w, h = w.int(), h.int()
            # cx, cy = (pos_anchor_box[:, 2] + pos_anchor_box[:, 0]) / 2, (pos_anchor_box[:, 3] + pos_anchor_box[:, 1]) / 2
            # x1, y1 = (cx - w / 2).int(), (cy - h / 2).int()
            # pos_anchor_num = pos_anchor_box.shape[0]
            # for j in range(pos_anchor_num):
            #     # trans_paste(w[j], h[j], color=color_list[label_target[j]], bg_img=image, alpha=pos_sample_pro[j]*0.8, box=(x1[j], y1[j]))
            #     trans_paste(w[j]-2, h[j]-2, color='green', bg_img=image, alpha=0.8, box=(x1[j]-1, y1[j]-1))
            #     # draw_bbox_text(drawobj, x1[j], y1[j], x1[j]+w[j], y1[j]+h[j], text='', color=color_list[label_target[j]], bd=1)

            plt.imshow(image)
            plt.savefig('debug/sample_pos.png')

            image = Image.fromarray(images_np[i])
            resized_h, resized_w, _ = img_meta['img_shape'][0]
            image = image.resize((resized_w, resized_h))
            drawobj = ImageDraw.ImageDraw(image)
            ignore_anchor_box_index = torch.nonzero((assigned_gt_inds_per_img == 0), as_tuple=False).view(-1)
            ignore_anchor_box = multi_level_anchors[ignore_anchor_box_index]
            w, h = (ignore_anchor_box[:, 2] - ignore_anchor_box[:, 0]) / self.anchor_generator.octave_base_scale, (ignore_anchor_box[:, 3] - ignore_anchor_box[:, 1]) / self.anchor_generator.octave_base_scale
            w, h = w.int(), h.int()
            cx, cy = (ignore_anchor_box[:, 2] + ignore_anchor_box[:, 0]) / 2, (ignore_anchor_box[:, 3] + ignore_anchor_box[:, 1]) / 2
            x1, y1 = (cx - w / 2).int(), (cy - h / 2).int()
            ignore_anchor_num = ignore_anchor_box.shape[0]
            for j in range(ignore_anchor_num):
                # trans_paste(w[j], h[j], color=color_list[label_target[j]], bg_img=image, alpha=pos_sample_pro[j]*0.8, box=(x1[j], y1[j]))
                trans_paste(w[j]-2, h[j]-2, color='blue', bg_img=image, alpha=0.8, box=(x1[j]-1, y1[j]-1))
            # image.show()
            # print(f"image showed")
            plt.imshow(image)
            plt.savefig('debug/sample_ignore.png')