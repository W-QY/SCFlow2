from typing import Optional, Dict, Sequence, Union
import torch
from torch import nn
from .base_flow_refiner import BaseFlowRefiner
from .builder import build_refiner



class BaseTeacherStudentFlowRefiner(BaseFlowRefiner):
    def __init__(self,
                student_model:dict, 
                renderer: dict,
                teacher_model:Optional[Dict]=None,
                max_flow: float = 400, 
                render_augmentations: Optional[Sequence] = None, 
                freeze_bn: bool = False, 
                train_cfg: dict = dict(), 
                test_cfg: dict = dict(), 
                init_cfg: dict = dict()):
        super().__init__(
            renderer=renderer, render_augmentations=render_augmentations, max_flow=max_flow, 
            train_cfg=train_cfg, test_cfg=test_cfg, init_cfg=init_cfg)
        student_model.update({'test_cfg':test_cfg})
        self.student = build_refiner(student_model)
        if teacher_model is not None:
            self.share_teacher = False
            self.teacher = build_refiner(teacher_model)
        else:
            self.share_teacher = True
            self.teacher = self.student
        
        if freeze_bn:
            self.freeze_bn(self.student)
            if not self.share_teacher:
                self.freeze_bn(self.teacher)
        
    
    def freeze_bn(self, model) -> None:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def to(self, device):
        super().to(device)
        self.teacher.to(device)
        self.student.to(device)


    def format_data_train(self, data_batch):
        ori_real_images, augmented_real_images = data_batch['ori_img'], data_batch['augmented_img']
        device = ori_real_images[0].device
        annots, meta_infos = data_batch['annots'], data_batch['img_metas']
        rotations_list, translations_list = annots['rotations'], annots['translations']
        gt_depths = annots.get('depths', [])
        gt_depths = torch.cat(gt_depths, dim=0) if len(gt_depths) != 0 else []
        
        supervise_data_flag = torch.tensor([True if 'gt_rotations' in meta_info else False for meta_info in meta_infos], dtype=torch.bool, device=device)
        labels, internel_k = annots['labels'], annots['k']
        labels, internel_k = torch.cat(labels), torch.cat(internel_k)
        rotations, translations = torch.cat(rotations_list), torch.cat(translations_list)
        image_num, view_num, _ = translations.shape

        augmented_real_images = torch.stack(augmented_real_images) # augmented_real_images:(N, V, C, H, W)
        ori_real_images = torch.stack(ori_real_images) # ori_real_images:(N, C, H, W)

        img_norm_cfg = meta_infos[0]['img_norm_cfg']
        normalize_mean, normalize_std = img_norm_cfg['mean'], img_norm_cfg['std']
        normalize_mean = torch.Tensor(normalize_mean).view(1, 3, 1, 1).to(augmented_real_images.device) / 255.
        normalize_std = torch.Tensor(normalize_std).view(1, 3, 1, 1).to(augmented_real_images.device) / 255.

        # format supervised data
        if supervise_data_flag.sum() > 0:
            gt_rotations = torch.cat([meta_info['gt_rotations'].data.to(device) for meta_info in meta_infos if 'gt_rotations' in meta_info])
            gt_translations = torch.cat([meta_info['gt_translations'].data.to(device) for meta_info in meta_infos if 'gt_translations' in meta_info])
            gt_masks = torch.cat([meta_info['gt_masks'].data.to_tensor(torch.bool, device) for meta_info in meta_infos if 'gt_masks' in meta_info])
            supervise_rotations, supervise_translations = rotations[supervise_data_flag][:, 0], translations[supervise_data_flag][:, 0]
            supervise_labels, supervise_internel_k = labels[supervise_data_flag], internel_k[supervise_data_flag]
            render_outputs = self.renderer(
                supervise_rotations, supervise_translations, 
                supervise_internel_k, supervise_labels,
            )
            rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
            rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
            rendered_depths = rendered_fragments.zbuf[..., 0]
            rendered_faces_index = rendered_fragments.pix_to_face[..., 0]
            rendered_masks = (rendered_depths > 0).to(torch.float32)
            rendered_images = (rendered_images - normalize_mean)/normalize_std
            if self.render_augmentation is not None:
                rendered_images = self.render_augmentation(rendered_images)
            supervise_data = dict(
                labels=supervise_labels, internel_k=supervise_internel_k,
                rotations=supervise_rotations, translations=supervise_translations,
                gt_rotations=gt_rotations, gt_translations=gt_translations, gt_masks=gt_masks,
                real_images=augmented_real_images[supervise_data_flag][:, 0], 
                rendered_images=rendered_images, rendered_masks=rendered_masks,
                rendered_depths=rendered_depths, rendered_faces_index=rendered_faces_index,
                image_num=supervise_data_flag.sum(),
            )

        # format unsupervised data
        rendered_depths_all, rendered_images_all, rendered_masks_all, rendered_face_index_all = [], [], [], []
        valid_gt_rotations_list, valid_gt_translations_list = [], []
        for i in range(image_num):
            if supervise_data_flag[i]:
                continue
            valid_gt_rotations_list.append(meta_infos[i]['valid_gt_rotations'].data.to(device))
            valid_gt_translations_list.append(meta_infos[i]['valid_gt_translations'].data.to(device))
            render_outputs = self.renderer(
                rotations[i], translations[i], 
                internel_k[i][None].expand(view_num, -1, -1), 
                labels[i][None].expand(view_num))
            rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
            rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
            rendered_depths = rendered_fragments.zbuf[..., 0]
            rendered_face_index = rendered_fragments.pix_to_face[..., 0]
            rendered_masks = (rendered_depths > 0).to(torch.float32)
            rendered_depths_all.append(rendered_depths)
            rendered_images_all.append(rendered_images)
            rendered_masks_all.append(rendered_masks)
            rendered_face_index_all.append(rendered_face_index)
        rendered_depths = torch.stack(rendered_depths_all, dim=0)
        rendered_masks = torch.stack(rendered_masks_all, dim=0)
        rendered_images = torch.stack(rendered_images_all, dim=0)
        rendered_faces_index = torch.stack(rendered_face_index_all, dim=0)
        valid_gt_rotations = torch.cat(valid_gt_rotations_list, dim=0)
        valid_gt_translations = torch.cat(valid_gt_translations_list, dim=0)
        if self.render_augmentation is not None:
            for i in range(image_num):
                rendered_images[i] = self.render_augmentation(rendered_images[i])
        rendered_images = (rendered_images - normalize_mean)/normalize_std
        unsupervise_data = dict(
                gt_depths=gt_depths,
                labels=labels[~supervise_data_flag], internel_k=internel_k[~supervise_data_flag],
                rotations=rotations[~supervise_data_flag], 
                translations=translations[~supervise_data_flag],
                gt_rotations=valid_gt_rotations,
                gt_translations=valid_gt_translations,
                ori_real_images=ori_real_images[~supervise_data_flag], 
                augmented_real_images=augmented_real_images[~supervise_data_flag],
                rendered_images=rendered_images, rendered_masks=rendered_masks, 
                rendered_depths=rendered_depths, rendered_faces_index=rendered_faces_index,
                image_num=(~supervise_data_flag).sum(), view_num=view_num, img_norm_cfg=img_norm_cfg
            )
        
        if supervise_data_flag.sum() == 0:
            return None, unsupervise_data
        elif torch.all(supervise_data_flag):
            return supervise_data, None
        else:
            return supervise_data, unsupervise_data
    
    def forward(self, data_batch, return_loss=False):
        data = self.format_data_test(data_batch)
        return self.student.forward_single_view(data, data_batch)