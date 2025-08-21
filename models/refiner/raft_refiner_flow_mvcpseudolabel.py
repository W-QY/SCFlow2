from typing import Optional, Sequence, Dict
import torch
from torch import nn
from torch.nn import functional as F
from .builder import REFINERS
from .base_teacher_student_flow_refiner import BaseTeacherStudentFlowRefiner
from collections import OrderedDict
from models.loss import build_loss
from models.utils import (
    Warp,
    cal_epe,
    filter_flow_by_mask,
    get_flow_from_delta_pose_and_depth)



@REFINERS.register_module()
class MVCRaftRefinerFlow(BaseTeacherStudentFlowRefiner):
    def __init__(self, 
                student_model: dict, 
                renderer: dict, 
                sup_flow_loss_cfg: Optional[dict]=None,
                selfsup_flow_loss_cfg: Optional[dict] = None,
                photometric_loss_cfg: Optional[dict] = None, 
                teacher_model: Optional[Dict] = None, 
                vis_tensorboard: bool=False,
                max_flow: float = 400, 
                render_augmentations: Optional[Sequence] = None, 
                freeze_bn: bool = False, 
                train_cfg: dict = dict(),
                test_cfg: dict = dict(),
                init_cfg: dict = dict()):
        super().__init__(
            student_model, renderer, teacher_model, max_flow, 
            render_augmentations, freeze_bn, train_cfg, test_cfg, init_cfg)
        self.sup_flow_loss_func = build_loss(sup_flow_loss_cfg) if sup_flow_loss_cfg else None
        self.selfsup_flow_loss_func = build_loss(selfsup_flow_loss_cfg) if selfsup_flow_loss_cfg else None
        self.photometric_loss_func = build_loss(photometric_loss_cfg) if photometric_loss_cfg else None
        self.vis_tensorboard = vis_tensorboard
        self.warp_op = Warp()
    
    def format_data_train(self, data_batch):
        supervise_data, unsupervise_data = super().format_data_train(data_batch)
        unsupervise_data['augmented_real_images'] = unsupervise_data['augmented_real_images'][:, 0]
        view_num = unsupervise_data['view_num']
        unsupervise_data['ori_real_images'] = unsupervise_data['ori_real_images'].expand(-1, view_num, -1, -1, -1)
        return supervise_data, unsupervise_data
    
    def loss(self, data_batch):
        supervise_data, unsupervise_data = self.format_data_train(data_batch)
        supervise_loss, supervise_log_imgs, supervise_log_vars = self.supervise_loss(supervise_data)
        unsupervise_loss, unsupervise_log_imgs, unsupervise_log_vars = self.unsupervise_loss(unsupervise_data)
        loss = supervise_loss + unsupervise_loss
        log_vars = OrderedDict()
        for k, v in supervise_log_vars.items():
            log_vars[k] = v 
        for k, v in unsupervise_log_vars.items():
            log_vars[k] = v 
        log_imgs = dict()
        for k, v in supervise_log_imgs.items():
            log_imgs[k] = v 
        for k, v in unsupervise_log_imgs.items():
            log_imgs[k] = v
        log_vars['loss'] = loss.item()
        return loss, log_imgs, log_vars

    def supervise_loss(self, data):
        if data is None:
            return 0., dict(), OrderedDict(),
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        ref_rotations, ref_translations = data['rotations'], data['translations']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        rendered_depths, rendered_masks = data['rendered_depths'], data['rendered_masks']
        internel_k, gt_masks = data['internel_k'], data['gt_masks']
        image_num = data['image_num']
        image_h, image_w = real_images.size(-2), real_images.size(-1)

        pred_seq_flow = self.student.get_flow(rendered_images, real_images)

        gt_flow = get_flow_from_delta_pose_and_depth(
            ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, 
            invalid_num=self.max_flow
        )
        gt_flow = filter_flow_by_mask(gt_flow, gt_masks, invalid_num=self.max_flow)
        loss_flow, seq_flow_loss_list = self.sup_flow_loss_func(
            pred_seq_flow, gt_flow=gt_flow, valid=rendered_masks, 
        )
        log_vars = OrderedDict()
        # for seq_i in range(len(seq_flow_loss_list)):
        #     log_vars[f'sup_seq_{seq_i}_flow_loss'] = seq_flow_loss_list[seq_i].item()
        log_vars['loss_sup_flow'] = loss_flow.item()
        if self.vis_tensorboard:
            pred_flow = pred_seq_flow[-1]
            pred_flow = pred_flow * rendered_masks[:, None]
            gt_flow = gt_flow.reshape(image_num, 2, image_h, image_w)
            # valid_mask = (gt_flow[:, :, 0] >= self.max_flow) & (gt_flow[:, :, 1] >= self.max_flow)
            # gt_flow = gt_flow * valid_mask[:, :, None]
            log_imgs = dict(
                pbr_pred_flow=pred_flow, pbr_gt_flow=gt_flow, 
                pbr_real_images=real_images, pbr_rendered_images=rendered_images,
                pbr_gt_masks=gt_masks.to(torch.float32)
            )
            log_imgs = self.add_vis_images(**log_imgs)
        return loss_flow, log_imgs, log_vars

    def unsupervise_loss(self, data):
        if data is None:
            return 0., dict(), dict()
        log_vars = OrderedDict()
        rotations, translations = data['rotations'], data['translations']
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        # ori_real_images:(N, C, H, W)
        ori_real_images, augmented_real_images = data['ori_real_images'], data['augmented_real_images']
        # rendered_images: (N, V, C, H, W)
        rendered_images, rendered_masks = data['rendered_images'], data['rendered_masks']
        rendered_depths, rendered_faces_index = data['rendered_depths'], data['rendered_faces_index']
        internel_k, labels = data['internel_k'], data['labels']
        image_num, view_num = data['image_num'], data['view_num']
        image_h, image_w = ori_real_images.size(-2), ori_real_images.size(-1)
        gt_depths = data['gt_depths']

        # student forward flow
        student_pred_seq_flow = self.student.get_flow(rendered_images[:, 0], augmented_real_images)
        loss = 0.
        
        # multiview teacher forward flow
        with torch.no_grad():
            teacher_pred_flow = self.teacher.get_flow(
                rendered_images.contiguous().view(-1, 3, image_h, image_w),
                ori_real_images.contiguous().view(-1, 3, image_h, image_w),
            )
            teacher_pred_flow = teacher_pred_flow[-1]   # (N × V, C, H, W)

        teacher_pred_flow = teacher_pred_flow.contiguous().view(image_num, view_num, 2, image_h, image_w)
        loss_selfsup, seq_loss_selfsup_list, flow_weights, mv_flow_var = self.selfsup_flow_loss_func(
            mv_teacher_pred_flow=teacher_pred_flow, sv_student_pred_flow=student_pred_seq_flow, 
            mv_rotations=rotations, mv_translations=translations, internel_k=internel_k,
            mv_rendered_depths=rendered_depths, mv_rendered_masks=rendered_masks, mv_rendered_faces_index=rendered_faces_index, gt_depths=gt_depths
        )
        log_vars['loss_selfsup_flow'] = loss_selfsup.item()
        loss += loss_selfsup

        # validation
        # import cv2
        # import numpy as np
        # gt_flow = get_flow_from_delta_pose_and_depth(
        #     rotations[:, 0], translations[:, 0], gt_rotations, gt_translations, 
        #     rendered_depths[:, 0], internel_k, invalid_num=self.max_flow
        # )
        # teacher_epe = cal_epe(gt_flow, teacher_pred_flow[:, 0], max_flow=self.max_flow, mask=rendered_masks[0, 0], reduction='none')
        # teacher_epe_vis = teacher_epe.cpu().numpy() / 5.
        # teacher_epe_vis = cv2.applyColorMap((teacher_epe_vis * 255).astype(np.uint8)[0], cv2.COLORMAP_JET)
        # cv2.imwrite(f'temp/train/53_168/teacher_epe_sv_100k.png', teacher_epe_vis)


        # base_path = 'temp/train/var_heat_51_1189'
        # import cv2 
        # from ..utils import Warp
        # from torchvision.utils import save_image
        # from mmcv.visualization import flow2rgb, flowshow
        # import numpy as np
        # warp_op = Warp()
        # warped_real_images = warp_op(ori_real_images[:, 0], student_pred_seq_flow[-1])
        # warped_real_images_r, warped_real_images_g, warped_real_images_b = warped_real_images[:, 0], warped_real_images[:, 1], warped_real_images[:, 2]
        # warped_real_images_r[rendered_masks[:, 0]==0] = 0.5
        # warped_real_images_g[rendered_masks[:, 0]==0] = 0.5
        # warped_real_images_b[rendered_masks[:, 0]==0] = 0.5
        # warped_real_images = torch.stack([warped_real_images_r, warped_real_images_g, warped_real_images_b], dim=1)
        # save_image(warped_real_images, f'{base_path}/warp_studendtflow.png', padding=0)
        # save_image(rendered_images[0], f'{base_path}/render.png', padding=0)

        # save_image(ori_real_images[:, 0], f'{base_path}/ori_real.png', padding=0)
        # save_image(augmented_real_images, f'{base_path}/aug_real.png', padding=0)
        
        # vis_teacher_flow = (teacher_pred_flow[0] * rendered_masks[0][:, None]).permute(0, 2, 3, 1).cpu().numpy()
        # vis_teacher_flow_list = []
        # for i in range(vis_teacher_flow.shape[0]): vis_teacher_flow_list.append(flow2rgb(vis_teacher_flow[i]))
        # vis_teacher_flow = cv2.hconcat(vis_teacher_flow_list)
        # cv2.imwrite(f'{base_path}/teacher_flow_sv_70k.png', (vis_teacher_flow*255).astype(np.uint8))

        # vis_teacher_flow = (teacher_pred_flow[0]).permute(0, 2, 3, 1).cpu().numpy()
        # vis_teacher_flow_list = []
        # for i in range(vis_teacher_flow.shape[0]): vis_teacher_flow_list.append(flow2rgb(vis_teacher_flow[i]))
        # vis_teacher_flow = cv2.hconcat(vis_teacher_flow_list)
        # cv2.imwrite(f'{base_path}/nomask_teacher_flow_sv_70k.png', (vis_teacher_flow*255).astype(np.uint8))

        # vis_student_pred_flow = (rendered_masks[:, 0][:, None] * student_pred_seq_flow[-1]).permute(0, 2, 3, 1).detach().cpu().numpy()
        # vis_student_pred_flow = flow2rgb(vis_student_pred_flow[0])
        # cv2.imwrite(f'{base_path}/student_flow.png', (vis_student_pred_flow*255).astype(np.uint8))
        


        if self.photometric_loss_func is not None:
            warped_real_image_list = [self.warp_op(ori_real_images[:, 0], f) for f in student_pred_seq_flow]
            loss_photometric, seq_loss_photometric_list = self.photometric_loss_func(
                warped_real_image_list, 
                src_images=rendered_images[:, 0], 
                valid=flow_weights,
                # valid=rendered_masks[:, 0],
                img_norm_cfg=data['img_norm_cfg'])
            # for i in range(len(seq_loss_photometric_list)):
            #     log_vars.update({f'seq_{i}_photo_loss':seq_loss_photometric_list[i].item()})
            log_vars.update({'loss_photometric':loss_photometric.item()})
            loss += loss_photometric
    
        # validation
        gt_flow = get_flow_from_delta_pose_and_depth(
            rotations[:, 0], translations[:, 0], gt_rotations, gt_translations, 
            rendered_depths[:, 0], internel_k, invalid_num=self.max_flow
        )
        teacher_epe = cal_epe(gt_flow, teacher_pred_flow[:, 0], max_flow=self.max_flow, mask=flow_weights, reduction='total_mean')
        for k, v in teacher_epe.items():
            log_vars['teacher_'+k] = v.item()
        student_epe = cal_epe(gt_flow, student_pred_seq_flow[-1], max_flow=self.max_flow, mask=flow_weights, reduction='total_mean')
        for k, v in student_epe.items():
            log_vars['student_'+k] = v.item()
        
        if False:
            teacher_epe = cal_epe(gt_flow, teacher_pred_flow[:, 0], max_flow=self.max_flow, mask=rendered_masks[:, 0], reduction='none')
            mv_flow_var[mv_flow_var == mv_flow_var.max()] = 0.
            from torchvision.utils import save_image
            save_image(teacher_epe[:, None]/10, 'debug_epe.png')
            save_image(mv_flow_var[:, None], 'debug_var.png')
            
    
        if self.vis_tensorboard:
            student_pred_flow = student_pred_seq_flow[-1].contiguous().view(image_num, 2, image_h, image_w)
            student_pred_flow = student_pred_flow * rendered_masks[:, 0, None]
            teacher_pred_flow = teacher_pred_flow.reshape(image_num, view_num, 2, image_h, image_w)
            teacher_pred_flow = teacher_pred_flow * rendered_masks[:, :, None]
            log_imgs = dict(
                    student_pred_flow=student_pred_flow, teacher_pred_flow=teacher_pred_flow, 
                    ori_real_image=ori_real_images[:, 0], augmented_real_image=augmented_real_images,
                    rendered_images=rendered_images, mvc_mask=flow_weights,
                )
            if self.photometric_loss_func is not None:
                log_imgs.update({'warped_images':warped_real_image_list[-1]*rendered_masks[:, 0, None]})
            else:
                warped_real_image_list = [self.warp_op(ori_real_images[:, 0], f) for f in student_pred_seq_flow]
                log_imgs.update({'warped_images':warped_real_image_list[-1]*rendered_masks[:, 0, None]})
            log_imgs = self.add_vis_images(**log_imgs)
            return loss, log_imgs, log_vars
        else:
            return loss, None, log_vars
    
    def train_step(self, data_batch, optimizer, **kwargs):
        loss, log_imgs, log_vars = self.loss(data_batch)
        outputs = dict(
            loss = loss,
            log_vars = log_vars,
            num_samples = len(data_batch['img_metas']),
        )
        if log_imgs is not None:
            outputs['log_imgs'] = log_imgs
        return outputs
