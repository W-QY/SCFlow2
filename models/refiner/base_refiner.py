import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Dict, Sequence
import mmcv
from mmcv.runner import BaseModule
import cv2, random, math, time
import numpy as np
from pathlib import Path
from kornia.augmentation import AugmentationSequential
from .builder import REFINERS
from ..encoder import build_encoder
from ..decoder import build_decoder
from ..utils import Renderer, build_augmentation, get_flow_from_delta_pose_and_depth, filter_flow_by_mask, cal_epe, RendererGSO
from ..utils.utils import simple_forward_warp, tensor_image_to_cv2, Warp, rgb_to_gray, dilate_mask_cuda
from ..utils import (
    solve_pose_by_pnp, get_2d_3d_corr_by_fw_flow,
    get_3d_3d_corr_by_fw_flow, solve_pose_by_ransac_kabsch)

@REFINERS.register_module()
class BaseRefiner(BaseModule):
    def __init__(self, 
                encoder: Optional[Dict]=None,
                decoder: Optional[Dict]=None,
                seperate_encoder: bool=False,
                renderer: Optional[Dict]=None,
                render_augmentations: Optional[Sequence[Dict]]=None,
                train_cfg: dict={},
                test_cfg: dict={},
                init_cfg: dict={},
                max_flow: int=400,
                ):
        super().__init__(init_cfg)
        self.seperate_encoder = seperate_encoder
        if encoder is not None:
            if self.seperate_encoder:
                self.render_encoder = build_encoder(encoder)
                self.real_encoder = build_encoder(encoder)
            else:
                encoder_model = build_encoder(encoder)
                self.render_encoder = encoder_model
                self.real_encoder = encoder_model
        if decoder is not None:
            self.decoder = build_decoder(decoder)  
        if renderer is not None: 
            if 'renderer_type' in renderer:
                self.renderer_type = 'GSO'
                self.renderer = RendererGSO(**renderer)
            else:
                self.renderer_type = None
                self.renderer = Renderer(**renderer)    
        else:
            self.renderer = None
        self.max_flow = max_flow
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.train_cycle_num = self.train_cfg.get('cycles', 1)
        self.train_grad_clip = self.train_cfg.get('grad_norm', None)
        self.test_cycle_num = self.test_cfg.get('cycles', 1)
        self.online_image_renderer = self.train_cfg.get('online_image_renderer', False)

        if render_augmentations is not None:
            augmentations = []
            for augmentation in render_augmentations:
                augmentations.append(
                    build_augmentation(augmentation)
                )
            self.render_augmentation = AugmentationSequential(
                *augmentations,
                data_keys=['input'],
                same_on_batch=False,
            )
        else:
            self.render_augmentation = None
        # self.measure_runtime()
    
    def measure_runtime(self):
        self.render_time = 0
        self.step = 0
    
    def update_measure_time(self, name, consume_time, steps=1):
        old_time = getattr(self, name)
        new_time = (old_time * self.step + consume_time*steps)/(self.step+steps)
        setattr(self, name, new_time)


    def to(self, device):
        super().to(device)
        if self.renderer is not None:
            self.renderer.to(device)
        
    def loss(self, data_batch):
        raise NotImplementedError
    
    def forward_single_view(self, data):
        raise NotImplementedError

    def format_data_test(self, data_batch):
        real_images, annots, meta_infos = data_batch['img'], data_batch['annots'], data_batch['img_metas']
        ref_rotations, ref_translations = annots['ref_rotations'], annots['ref_translations']
        labels, internel_k = annots['labels'], annots['k']
        ori_k, transform_matrixs = annots['ori_k'], annots['transform_matrix']

        per_img_patch_num = [len(images) for images in real_images]
        real_images = torch.cat(real_images)
        ref_rotations, ref_translations = torch.cat(ref_rotations, dim=0), torch.cat(ref_translations, dim=0)
        labels = torch.cat(labels)
        internel_k = torch.cat(internel_k)
        transform_matrixs = torch.cat(transform_matrixs)
        ori_k = torch.cat([k[None].expand(patch_num, 3, 3) for k, patch_num in zip(ori_k, per_img_patch_num)])
        render_outputs = self.renderer(ref_rotations, ref_translations, internel_k, labels)
        rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
        if self.test_cfg.get('rgb_to_gary', False):
            rendered_images = rgb_to_gray(rendered_images)
        rendered_depths = rendered_fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        rendered_masks = (rendered_depths > 0).to(torch.float32)
        if self.train_cfg.get('rendered_mask_filte', False):
            dilate_mask = dilate_mask_cuda(rendered_masks)
            real_images = real_images * dilate_mask.unsqueeze(1)
        img_norm_cfg = meta_infos[0]['img_norm_cfg']
        normalize_mean, normalize_std = img_norm_cfg['mean'], img_norm_cfg['std']
        normalize_mean = torch.Tensor(normalize_mean).view(1, 3, 1, 1).to(real_images[0].device) / 255.
        normalize_std = torch.Tensor(normalize_std).view(1, 3, 1, 1).to(real_images[0].device) / 255.
        rendered_images = (rendered_images - normalize_mean)/normalize_std
        output = dict(
            real_images = real_images,
            rendered_images = rendered_images,
            labels = labels,
            ori_k = ori_k,
            transform_matrix = transform_matrixs,
            internel_k = internel_k,
            ref_rotations = ref_rotations,
            ref_translations = ref_translations,
            rendered_masks = rendered_masks,
            rendered_depths = rendered_depths,  
            per_img_patch_num = per_img_patch_num,
            meta_infos=meta_infos,
        )
        if 'cloud_list' in annots:
            output['cloud_list'] = torch.cat(annots['cloud_list'], dim=0)
        if 'model_list' in annots:
            output['model_list'] = torch.cat(annots['model_list'], dim=0)
        if 'depths' in annots:
            real_depths = torch.cat(annots['depths'], dim=0)
            if self.train_cfg.get('rendered_mask_filte', False):
                real_depths = real_depths * dilate_mask
            output.update(real_depths=real_depths)
        else:
            real_depths = []
            output.update(real_depths=real_depths)
        if 'gt_rotations' in annots:
            gt_rotations, gt_translaions = annots['gt_rotations'], annots['gt_translations']
            gt_rotations, gt_translaions = torch.cat(gt_rotations, dim=0), torch.cat(gt_translaions, dim=0)
            output.update(
                gt_rotations=gt_rotations,
                gt_translations=gt_translaions,
            )
        if 'gt_masks' in annots:
            gt_masks = [mask.to_tensor(dtype=torch.bool, device=real_images[0].device) for mask in annots['gt_masks']]
            gt_masks = torch.cat(gt_masks, axis=0).to(torch.float32)
            output.update(gt_masks=gt_masks)
        return output


    def format_data_train_sup(self, data_batch):
        real_images, annots, meta_infos = data_batch['img'], data_batch['annots'], data_batch['img_metas']
        gt_rotations, gt_translations = annots['gt_rotations'], annots['gt_translations']
        ref_rotations, ref_translations = annots['ref_rotations'], annots['ref_translations']
        init_add_error, init_rot_error, init_trans_error = annots['init_add_error'], annots['init_rot_error'], annots['init_trans_error']
        labels, internel_k = annots['labels'], annots['k']
        init_rot_error_std, init_rot_error_mean = torch.std_mean(init_rot_error, unbiased=False)
        init_add_error_std, init_add_error_mean = torch.std_mean(init_add_error, unbiased=False)
        init_trans_error_std, init_trans_error_mean = torch.std_mean(init_trans_error, unbiased=False)

        real_images = torch.cat(real_images)
        ref_rotations, ref_translations = torch.cat(ref_rotations, axis=0), torch.cat(ref_translations, axis=0)
        gt_rotations, gt_translations = torch.cat(gt_rotations, axis=0), torch.cat(gt_translations, axis=0)
        labels, internel_k = torch.cat(labels), torch.cat(internel_k)
        if self.online_image_renderer:
            rendered_depths = torch.squeeze(data_batch['annots']['rendered_depths'], dim=1)
            rendered_images = torch.squeeze(data_batch['annots']['rendered_images'], dim=1)
            rendered_masks = torch.squeeze(data_batch['annots']['rendered_masks'], dim=1)
            # render_outputs = self.renderer(ref_rotations, ref_translations, internel_k, labels)
        else:
            render_outputs = self.renderer(ref_rotations, ref_translations, internel_k, labels)
            rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
            rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
            rendered_depths = rendered_fragments.zbuf
            rendered_depths = rendered_depths[..., 0]
            rendered_masks = (rendered_depths > 0).to(torch.float32)
            if self.render_augmentation is not None:
                rendered_images = self.render_augmentation(rendered_images)
            img_norm_cfg = meta_infos[0]['img_norm_cfg']
            normalize_mean, normalize_std = img_norm_cfg['mean'], img_norm_cfg['std']
            normalize_mean = torch.Tensor(normalize_mean).view(1, 3, 1, 1).to(real_images[0].device) / 255.
            normalize_std = torch.Tensor(normalize_std).view(1, 3, 1, 1).to(real_images[0].device) / 255.
            rendered_images = (rendered_images - normalize_mean)/normalize_std
        if self.train_cfg.get('rendered_mask_filte', False):
            dilate_mask = dilate_mask_cuda(rendered_masks)
            real_images = real_images * dilate_mask.unsqueeze(1)
        output = dict(
            ref_rotations = ref_rotations,
            ref_translations = ref_translations,
            gt_rotations = gt_rotations,
            gt_translations = gt_translations,
            labels = labels,
            internel_k = internel_k,
            rendered_images = rendered_images,
            real_images = real_images,
            rendered_masks = rendered_masks,
            rendered_depths = rendered_depths,
            init_add_error_mean = init_add_error_mean,
            init_add_error_std = init_add_error_std,
            init_rot_error_mean = init_rot_error_mean,
            init_rot_error_std = init_rot_error_std,
            init_trans_error_mean = init_trans_error_mean,
            init_trans_error_std = init_trans_error_std,
        )
        if 'cloud_list' in annots:
            output['cloud_list'] = torch.cat(annots['cloud_list'])
        if 'model_list' in annots:
            output['model_list'] = torch.cat(annots['model_list'])
        if 'model_eval_list' in annots:
            output['model_eval_list'] = torch.cat(annots['model_eval_list'])
        if 'models_diameter' in annots:
            output['models_diameter'] = torch.cat(annots['models_diameter'])
        if 'depths' in annots:
            gt_depths = annots['depths']
            gt_depths = torch.cat(gt_depths)
            if self.train_cfg.get('rendered_mask_filte', False):
                gt_depths = gt_depths * dilate_mask
            output['gt_depths'] = gt_depths
        else:
            output['gt_depths'] = []
        if 'gt_masks' in annots:
            gt_masks = [mask.to_tensor(dtype=torch.bool, device=gt_rotations[0].device) for mask in annots['gt_masks']]
            gt_masks = torch.cat(gt_masks, axis=0)
            if self.train_cfg.get('rendered_mask_filte', False):
                gt_masks = gt_masks * dilate_mask
            output['gt_masks'] = gt_masks
            return output
        else:
            return output


    def format_data_train_sup_mhp(self, data_batch, mhp_num):
        def duplicate_in_place(lst):
            k, n = int(mhp_num), len(lst)
            lst.extend([None] * (n * (k - 1)))
            for i in range(n - 1, -1, -1):
                target_idx = i * k
                for j in range(k):
                    lst[target_idx + j] = lst[i]
            return torch.cat(lst, axis=0)
        def merge_and_cat(lst):
            merged_tensors = [t.reshape(-1, *t.shape[2:]) for t in lst]
            return torch.cat(merged_tensors, dim=0)

        real_images, annots, meta_infos = duplicate_in_place(data_batch['img']), data_batch['annots'], data_batch['img_metas']
        gt_rotations, gt_translations = duplicate_in_place(annots['gt_rotations']), duplicate_in_place(annots['gt_translations'])
        labels, internel_k = duplicate_in_place(annots['labels']), duplicate_in_place(annots['k'])
        model_eval_list, models_diameter = duplicate_in_place(annots['model_eval_list']), duplicate_in_place(annots['models_diameter'])
        gt_masks = [mask.to_tensor(dtype=torch.bool, device=gt_rotations[0].device) for mask in annots['gt_masks']]
        gt_depths, gt_masks = duplicate_in_place(annots['depths']), duplicate_in_place(gt_masks)

        ref_rotations, ref_translations = merge_and_cat(annots['ref_rotations']), merge_and_cat(annots['ref_translations'])
        rendered_depths = annots['rendered_depths'].reshape(-1, *annots['rendered_depths'].shape[2:])
        rendered_images = annots['rendered_images'].reshape(-1, *annots['rendered_images'].shape[2:])
        rendered_masks = annots['rendered_masks'].reshape(-1, *annots['rendered_masks'].shape[2:])

        init_add_error, init_rot_error, init_trans_error = annots['init_add_error'], annots['init_rot_error'], annots['init_trans_error']
        init_rot_error_std, init_rot_error_mean = torch.std_mean(init_rot_error, unbiased=False)
        init_add_error_std, init_add_error_mean = torch.std_mean(init_add_error, unbiased=False)
        init_trans_error_std, init_trans_error_mean = torch.std_mean(init_trans_error, unbiased=False)

        if self.train_cfg.get('rendered_mask_filte', False):
            dilate_mask = dilate_mask_cuda(rendered_masks)
            real_images = real_images * dilate_mask.unsqueeze(1)
            gt_depths = gt_depths * dilate_mask
            gt_masks = gt_masks * dilate_mask
        
        output = dict(
            ref_rotations = ref_rotations,
            ref_translations = ref_translations,
            gt_rotations = gt_rotations,
            gt_translations = gt_translations,
            gt_depths = gt_depths,
            gt_masks = gt_masks,
            labels = labels,
            internel_k = internel_k,
            rendered_images = rendered_images,
            real_images = real_images,
            rendered_masks = rendered_masks,
            rendered_depths = rendered_depths,
            model_eval_list = model_eval_list, 
            models_diameter = models_diameter,
            init_add_error_mean = init_add_error_mean,
            init_add_error_std = init_add_error_std,
            init_rot_error_mean = init_rot_error_mean,
            init_rot_error_std = init_rot_error_std,
            init_trans_error_mean = init_trans_error_mean,
            init_trans_error_std = init_trans_error_std,
        )
        return output


    def format_data_test_mhp(self, data_batch, mhp_num):
        def duplicate_in_place(lst):
            return torch.cat(lst).repeat_interleave(mhp_num, dim=0)

        def merge_and_cat(lst):
            merged_tensors = [t.reshape(-1, *t.shape[2:]) for t in lst]
            return torch.cat(merged_tensors, dim=0)

        real_images, annots, meta_infos = data_batch['img'], data_batch['annots'], data_batch['img_metas']
        per_img_patch_num = [len(images) for images in real_images]
        real_images = duplicate_in_place(real_images)
        labels, internel_k = duplicate_in_place(annots['labels']), duplicate_in_place(annots['k'])
        gt_depths = duplicate_in_place(annots['depths'])

        ori_k, transform_matrixs = annots['ori_k'], annots['transform_matrix']
        ori_k = torch.cat([k[None].expand(patch_num, 3, 3) for k, patch_num in zip(ori_k, per_img_patch_num)])
        transform_matrixs = torch.cat(transform_matrixs)

        ref_rotations, ref_translations = merge_and_cat(annots['ref_rotations']), merge_and_cat(annots['ref_translations'])

        render_outputs = self.renderer(ref_rotations, ref_translations, internel_k, labels)
        rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
        if self.test_cfg.get('rgb_to_gary', False):
            rendered_images = rgb_to_gray(rendered_images)
        rendered_depths = rendered_fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        rendered_masks = (rendered_depths > 0).to(torch.float32)
        if self.train_cfg.get('rendered_mask_filte', False):
            dilate_mask = dilate_mask_cuda(rendered_masks)
            real_images = real_images * dilate_mask.unsqueeze(1)
        img_norm_cfg = meta_infos[0]['img_norm_cfg']
        normalize_mean, normalize_std = img_norm_cfg['mean'], img_norm_cfg['std']
        normalize_mean = torch.Tensor(normalize_mean).view(1, 3, 1, 1).to(real_images[0].device) / 255.
        normalize_std = torch.Tensor(normalize_std).view(1, 3, 1, 1).to(real_images[0].device) / 255.
            # rendered_images = (rendered_images - normalize_mean)/normalize_std
            # rendered_depths_all.append(rendered_depths)
            # rendered_images_all.append(rendered_images)
            # rendered_masks_all.append(rendered_masks)
        # rendered_depths = torch.stack(rendered_depths_all, dim=0)
        # rendered_masks = torch.stack(rendered_masks_all, dim=0)
        # rendered_images = torch.stack(rendered_images_all, dim=0)



        if self.train_cfg.get('rendered_mask_filte', False):
            dilate_mask = dilate_mask_cuda(rendered_masks)
            real_images = real_images * dilate_mask.unsqueeze(1)
            gt_depths = gt_depths * dilate_mask


        output = dict(
            real_images = real_images,
            rendered_images = rendered_images,
            labels = labels,
            ori_k = ori_k,
            real_depths = gt_depths,
            transform_matrix = transform_matrixs,
            internel_k = internel_k,
            ref_rotations = ref_rotations,
            ref_translations = ref_translations,
            rendered_masks = rendered_masks,
            rendered_depths = rendered_depths,  
            per_img_patch_num = per_img_patch_num,
            meta_infos=meta_infos,
        )

        # if 'depths' in annots:
        #     real_depths = torch.cat(annots['depths'], dim=0)
        #     if self.train_cfg.get('rendered_mask_filte', False):
        #         real_depths = real_depths * dilate_mask
        #     output.update(real_depths=real_depths)
        # else:
        #     real_depths = []
        #     output.update(real_depths=real_depths)
        # if 'gt_masks' in annots:
        #     gt_masks = [mask.to_tensor(dtype=torch.bool, device=gt_rotations[0].device) for mask in annots['gt_masks']]
        #     gt_masks = torch.cat(gt_masks, axis=0)
        #     output.update(gt_masks=gt_masks)
        if 'gt_rotations' in annots:
            gt_rotations, gt_translaions = annots['gt_rotations'], annots['gt_translations']
            gt_rotations, gt_translaions = torch.cat(gt_rotations, dim=0), torch.cat(gt_translaions, dim=0)
            output.update(
                gt_rotations=gt_rotations,
                gt_translations=gt_translaions,
            )
        return output

    
    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad_norm = nn.utils.clip_grad.clip_grad_norm_(params, **self.train_grad_clip)
            return grad_norm

    
    def update_data(self, update_rotations, update_translations, data):
        data['ref_rotations'] = update_rotations
        data['ref_translations'] = update_translations
        labels, internel_k = data['labels'], data['internel_k']
        render_outputs = self.renderer(update_rotations, update_translations, internel_k, labels)
        rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
        rendered_depths = rendered_fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        rendered_masks = (rendered_depths > 0).to(torch.float32)
        data['rendered_images'] = rendered_images
        data['rendered_depths'] = rendered_depths
        data['rendered_masks'] = rendered_masks
        return data

    def train_multiple_iterations(self, data_batch, optimizer):
        train_cycle_num = getattr(self, 'train_cycle_num')
        iter_loss_dict = dict()
        data = self.format_data_train_sup(data_batch)
        log_vars_list, log_imgs_list = [], []
        for i in range(train_cycle_num):
            loss, log_imgs, log_vars, seq_rotations, seq_translations = self.loss(data, data_batch)
            log_vars_list.append(log_vars)
            log_imgs_list.append(log_imgs)

            if i == train_cycle_num - 1:
                continue
            
            optimizer.zero_grad()
            loss.backward()
            self.clip_grads(self.parameters())
            optimizer.step()

            update_rotations, update_translations = seq_rotations[-1], seq_translations[-1]
            update_rotations = update_rotations.detach()
            update_translations = update_translations.detach()
            data = self.update_data(update_rotations, update_translations, data)

            iter_loss_dict[f'iter_{i}_loss'] = loss.item()
        log_vars = {k:sum([log_vars[k] for log_vars in log_vars_list])/train_cycle_num for k in log_vars_list[0]}
        log_vars.update(iter_loss_dict)
        log_imgs = log_imgs_list[random.choice(list(range(train_cycle_num-1)))]
        return loss, log_imgs, log_vars
    
    def forward_multiple_pass(self, data):
        for i in range(self.test_cycle_num):
            results_dict = self.forward_single_pass(data)
            if i == self.test_cycle_num - 1:
                continue
            update_rotations = results_dict['rotations']
            update_translations = results_dict['translations']
            update_rotations = torch.cat(update_rotations)
            update_translations = torch.cat(update_translations)
            # start = time.time()
            data = self.update_data(update_rotations, update_translations, data)
            # render_time = (time.time() - start)/len(runtime_total)
            # runtime_total = [r_1+render_time for r_1 in runtime_total]
        batch_size = len(data['real_images'])
        # self.runtime_record.extend(runtime_total)
        # self.report_runtime(new_step=len(runtime_total))
        return results_dict


    def vis_flow(self, val):
        if isinstance(val, (list, tuple)):
            # sequence preiction
            # visualize the same sample's prediction across different iterations
            flow_list = []
            for i in range(len(val)):
                flow_i = val[i][0].permute(1, 2, 0).cpu().data.numpy()
                flow_i = mmcv.flow2rgb(flow_i, unknown_thr=self.max_flow-1)
                flow_list.append(flow_i)
            return flow_list
        else:
            assert isinstance(val, torch.Tensor)
            if val.ndim == 4:
                flow = val[0].permute(1, 2, 0).cpu().data.numpy()
                flow = mmcv.flow2rgb(flow, unknown_thr=self.max_flow-1)
                return flow
            elif val.ndim == 5:
                flow_list = []
                for i in range(val.size(1)):
                    flow = val[0, i].permute(1, 2, 0).cpu().data.numpy()
                    flow = mmcv.flow2rgb(flow, unknown_thr=self.max_flow-1)
                    flow_list.append(flow)
                return flow_list
            else:
                raise RuntimeError
    
    def vis_images(self, val:torch.Tensor):
        if val.ndim == 4:
            return val[0].permute(1, 2, 0).cpu().data.numpy()
        elif val.ndim == 5:
            image_list = []
            for i in range(val.size(1)):
                image = val[0, i].permute(1, 2, 0).cpu().data.numpy()
                image_list.append(image)
            return image_list
    
    def vis_masks(self, val:torch.Tensor):
        if val.ndim == 3:
            return val[0, None].permute(1, 2, 0).cpu().data.numpy()
        elif val.ndim == 4:
            mask_list = []
            for i in range(val.size(1)):
                mask = val[0, i][None].permute(1, 2, 0).cpu().data.numpy()
                mask_list.append(mask)
            return mask_list
            
    def add_vis_images(self, **kwargs):
        log_imgs = dict()
        
        for key, val in kwargs.items():
            if 'flow' in key:
                log_imgs[key] = self.vis_flow(val)
            elif 'image' in key:
                log_imgs[key] = self.vis_images(val)
            elif 'mask' in key:
                log_imgs[key] = self.vis_masks(val)
            else:
                raise RuntimeError
        return log_imgs
    
    def train_step(self, data_batch, optimizer, **kwargs):
        if self.train_cycle_num > 1:
            loss, log_imgs, log_vars = self.train_multiple_iterations(data_batch, optimizer)
        else:
            loss, log_imgs, log_vars, _, _ = self.loss(data_batch)
        outputs = dict(
            loss = loss,
            log_vars = log_vars,
            log_imgs = log_imgs,
            num_samples = len(data_batch['img_metas']),
        )
        return outputs
    
    def forward(self, data_batch, return_loss=False, fdpose=False):
        if fdpose:
            return self.forward_fdpose(data_batch)
        data = self.format_data_test(data_batch)
        if self.test_cycle_num > 1:
            return self.forward_multiple_pass(data)
        else:
            return self.forward_single_pass(data, data_batch)
    
    
    def visualize_sequence_flow_and_fw(self, data, sequence_flow):
        output_dir = self.test_cfg.get('vis_dir')
        meta_infos = data['meta_infos']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        per_img_patch_num = data['per_img_patch_num']
        rendered_depths, rendered_masks = data['rendered_depths'], data['rendered_masks']
        internel_k, labels = data['internel_k'], data['labels']
        batchsize = len(real_images)
        show_index = self.test_cfg.get('vis_index', None)
        if show_index is None:
            show_index = range(len(sequence_flow))
        flow_list = [f for j, f in enumerate(sequence_flow) if j in show_index]
        real_images_cv2 = tensor_image_to_cv2(real_images)
        fw_batch_images = [
            tensor_image_to_cv2(simple_forward_warp(rendered_images, f, rendered_masks))
            for f in flow_list
        ]
        image_index = 0
        show_image_list_all = []
        for i in range(batchsize):
            meta_info = meta_infos[image_index]
            sequence = str(Path(meta_info['img_path']).parents[1].name)
            fw_image = [fw_batch_image[i] for fw_batch_image in fw_batch_images]

            flow_image = [
                (mmcv.flow2rgb((flow[i]*rendered_masks[i][None]).permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)[..., ::-1]*255).astype(np.uint8)
                for flow in flow_list]
            
            # diff_image = [
            #     np.abs(real_images_cv2[i] - fw_image_j)
            #     for fw_image_j in fw_image
            # ]
            show_image_list = []
            for j in range(len(flow_image)):
                show_image_list.append(flow_image[j])
                show_image_list.append(fw_image[j])
                # show_image_list.append(diff_image[j])

            show_image = np.concatenate(show_image_list, axis=1)
            show_image_list_all.append(show_image)
            if i >= sum(per_img_patch_num[:image_index+1])-1:
                image_index += 1
                show_image_all = np.concatenate(show_image_list_all, axis=0)
                save_path = Path(output_dir).joinpath(sequence + '_'+str(Path(meta_info['img_path']).stem) + "_flow.png")
                mmcv.mkdir_or_exist(Path(save_path).parent)
                cv2.imwrite(save_path.as_posix(), show_image_all)
                show_image_list_all = []






    def visualize_and_save(self, data, sequence_flow, pred_rotations, pred_translations, sequence_pose_flow=None):
        output_dir = self.test_cfg.get('vis_dir')
        meta_infos = data['meta_infos']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        per_img_patch_num = data['per_img_patch_num']
        rendered_depths, rendered_masks = data['rendered_depths'], data['rendered_masks']
        internel_k, labels = data['internel_k'], data['labels']
        gt_masks = data.get("gt_masks", None)
        real_images_cv2 = tensor_image_to_cv2(real_images)
        rendered_images_cv2 = tensor_image_to_cv2(rendered_images)
        batchsize = len(real_images)
        image_index = 0
        render_outputs = self.renderer(pred_rotations, pred_translations, internel_k, labels)
        refined_images, refined_fragments = render_outputs['images'], render_outputs['fragments']
        refined_images = refined_images[..., :3].permute(0, 3, 1, 2).contiguous()
        refined_depths = refined_fragments.zbuf
        refined_depths = refined_depths[..., 0]
        refined_masks = (refined_depths > 0).to(torch.float32)

        show_index = self.test_cfg.get('vis_pose_index', -1)

        if 'gt_rotations' in data:
            gt_flow = get_flow_from_delta_pose_and_depth(data['ref_rotations'], data['ref_translations'], data['gt_rotations'], data['gt_translations'], rendered_depths, internel_k, invalid_num=0)
            warp_image_by_gt_flow = simple_forward_warp(rendered_images, gt_flow, rendered_masks)
            warp_image_by_gt_flow = tensor_image_to_cv2(warp_image_by_gt_flow)
        else:
            gt_flow = None

        if sequence_pose_flow is not None:
            # needs flow shape (B, H, W, 2)
            warp_image_by_pose_flow = simple_forward_warp(rendered_images, sequence_pose_flow[show_index], rendered_masks)
            # warp_image_by_pose_flow = forward_warp_func(rendered_images, sequence_pose_flow[-1].permute(0, 2, 3, 1))
            warp_image_by_pose_flow = tensor_image_to_cv2(warp_image_by_pose_flow)
        warp_image_by_pred_flow = simple_forward_warp(rendered_images, sequence_flow[show_index], rendered_masks)
        warp_image_by_pred_flow = tensor_image_to_cv2(warp_image_by_pred_flow)
        show_image_list_all = []
        for i in range(batchsize):
            show_image_list = [real_images_cv2[i], rendered_images_cv2[i]]
            # flow = np.concatenate([mmcv.flow2rgb(flow[i].permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)[..., ::-1]*255 for flow in sequence_flow], axis=1)
            # pose_flow = np.concatenate([mmcv.flow2rgb(flow[i].permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)[..., ::-1]*255 for flow in sequence_pose_flow], axis=1)
            meta_info = meta_infos[image_index]
            sequence = str(Path(meta_info['img_path']).parents[1].name)
            patch_index = i - sum(per_img_patch_num[:image_index])
            # save_path = Path(output_dir).joinpath(sequence).joinpath(str(Path(meta_info['img_path']).stem) + f"_{patch_index:06d}.png")
            # mmcv.mkdir_or_exist(Path(save_path).parent)
            flow = mmcv.flow2rgb((sequence_flow[show_index][i]*rendered_masks[i][None]).permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)
            flow = (flow[..., ::-1]*255).astype(np.uint8)
            show_image_list.append(flow)
            show_image_list.append(warp_image_by_pred_flow[i])
            if sequence_pose_flow is not None:
                pose_flow = mmcv.flow2rgb(sequence_pose_flow[show_index][i].permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)
                pose_flow = (pose_flow[..., ::-1]*255).astype(np.uint8)
                show_image_list.append(pose_flow)
                show_image_list.append(warp_image_by_pose_flow[i])

            if gt_flow is not None:
                gt_flow_i = gt_flow[i]
                gt_flow_i = mmcv.flow2rgb(gt_flow_i.permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)
                gt_flow_i = (gt_flow_i[..., ::-1]*255).astype(np.uint8)
                show_image_list.append(gt_flow_i)
                show_image_list.append(warp_image_by_gt_flow[i])

            
            if gt_masks is not None:
                contours_gt, _ = cv2.findContours(gt_masks[i].cpu().data.numpy().astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            contours_init, _ = cv2.findContours(rendered_masks[i].cpu().data.numpy().astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            ref_pose_show = cv2.drawContours(real_images_cv2[i].copy(), contours_init, -1, [0, 0, 255], 3)          # red
            ref_pose_show = cv2.drawContours(ref_pose_show.copy(), contours_gt, -1, [255, 0, 0], 3)                 # blue
            show_image_list.append(ref_pose_show)
            contours_refined, _ = cv2.findContours(refined_masks[i].cpu().data.numpy().astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            #-# modified in 20250312
            # refine_pose_show = cv2.drawContours(real_images_cv2[i].copy(), contours_refined, -1, [255, 255, 255], 3)
            refine_pose_show = cv2.drawContours(real_images_cv2[i].copy(), contours_refined, -1,  [0, 255, 0], 3)   # green
            # refine_pose_show = cv2.drawContours(refine_pose_show.copy(), contours_init, -1, [0, 0, 255], 3)         # red
            refine_pose_show = cv2.drawContours(refine_pose_show.copy(), contours_gt, -1, [255, 0, 0], 3)           # blue
            
            
            show_image_list.append(refine_pose_show)
            show_image = np.concatenate(show_image_list, axis=1)
            show_image_list_all.append(show_image)
            # cv2.imwrite(save_path.as_posix(), show_image)

            if i >= sum(per_img_patch_num[:image_index+1])-1:
                image_index += 1
                show_image_all = np.concatenate(show_image_list_all, axis=0)
                save_path = Path(output_dir).joinpath(sequence + '_'+str(Path(meta_info['img_path']).stem) + ".png")
                mmcv.mkdir_or_exist(Path(save_path).parent)
                cv2.imwrite(save_path.as_posix(), show_image_all)
                show_image_list_all = []
    
    def eval_seq_epe(self, sequence_flow, rendered_depths, ref_rotations, ref_translations, internel_k, gt_rotations, gt_translations, render_masks, gt_masks=None):
        sequence_epe = []
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        for i in range(len(sequence_flow)):
            flow_error = torch.sum((gt_flow - sequence_flow[i])**2, dim=1).sqrt()
            flow_error = flow_error * render_masks
            sequence_epe.append(flow_error)
        for i in range(len(sequence_flow[0])):
            epe_list = [s[i] for s in sequence_epe]
            epe_mean = [torch.sum(epe)/render_masks[i].sum() for epe in epe_list]
            print(epe_mean)
            epe = torch.cat(epe_list, dim=1)
            epe = (epe/epe.max()).cpu().data.numpy()
            # np.save(f'debug/flow_{i}.npy', epe)

            epe = (epe*255).astype(np.uint8)
            epe = cv2.applyColorMap(epe[..., None], cv2.COLORMAP_JET)
            cv2.imwrite(f'debug/flow_{i}.png', epe)          

    
    def eval_similarity(self, feat_render, feat_real, rendered_depths, ref_rotations, ref_translations, internel_k, gt_rotations, gt_translations,gt_masks=None, similarity_mode='cosine', scale=8):
        assert similarity_mode in ['cosine', 'absolute', 'relative']
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        if gt_masks is not None:
            gt_flow = filter_flow_by_mask(gt_flow, gt_masks, self.max_flow)
        valid_mask = torch.sum(gt_flow**2, dim=1).sqrt() < self.max_flow
        warp_func = Warp()
        downsampled_gt_flow = 1/scale * F.interpolate(
                gt_flow, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True) 
        valid_mask = valid_mask[:, None].to(torch.float32)
        downsampled_valid_mask = F.interpolate(
            valid_mask, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True)[:, 0, ...]
        warped_feat = warp_func(feat_real, downsampled_gt_flow)
        if similarity_mode == 'cosine':
            similarity = torch.cosine_similarity(feat_render, warped_feat, dim=1)
        elif similarity_mode == 'absolute':
            similarity = (feat_real * feat_render).sum(dim=1)/math.sqrt(feat_render.size(1))
        else:
            N, C, H, W = feat_real.size()
            similarity = (feat_real * feat_render).sum(dim=1)/torch.sqrt(torch.tensor(C).float())
            corr = torch.matmul(feat_render.view(N, C, -1).permute(0, 2, 1), feat_real.view(N, C, -1)).view(N, H, W, H, W)
            corr = corr / torch.sqrt(torch.tensor(C).float())
            # corr = torch.exp(corr).sum(dim=(-1, -2))
            # similarity = torch.exp(similarity) / corr
            corr = torch.amax(corr, dim=(-1, -2))
            similarity = similarity / corr

        similarity = similarity * downsampled_valid_mask
        similarity_map = similarity
        similarity = similarity.sum(dim=(-1, -2)) /(downsampled_valid_mask.sum(dim=(-1, -2)) + 1e-8)
        return similarity, similarity_map
    
    def vis_similarity(self, similarity, data, scale=8):
        output_dir = self.test_cfg.get('vis_dir')
        meta_infos = data['meta_infos']
        per_img_patch_num = data['per_img_patch_num']
        real_images = data['real_images']
        rendered_images = data['rendered_images']
        real_images_cv2 = tensor_image_to_cv2(real_images)
        rendered_images_cv2 = tensor_image_to_cv2(rendered_images)
        batch_size = len(similarity)
        similarity = F.interpolate(similarity[:, None], scale_factor=(scale, scale), mode='bilinear', align_corners=True)[:, 0]
        similarity = (similarity * 255).clamp(0, 255).byte().cpu().numpy()
        image_index = 0
        show_image_list_all = []
        for i in range(batch_size):
            show_image_list = [real_images_cv2[i], rendered_images_cv2[i]]
            meta_info = meta_infos[image_index]
            sequence = str(Path(meta_info['img_path']).parents[1].name)
            save_similarity = cv2.applyColorMap(similarity[i], cv2.COLORMAP_JET)
            show_image_list.append(save_similarity)
            show_image = np.concatenate(show_image_list, axis=1)
            show_image_list_all.append(show_image)
            if i >= sum(per_img_patch_num[:image_index+1])-1:
                image_index += 1
                show_image_all = np.concatenate(show_image_list_all, axis=0)
                save_path = Path(output_dir).joinpath(sequence).joinpath(str(Path(meta_info['img_path']).stem) + "_sim_2.png")
                mmcv.mkdir_or_exist(Path(save_path).parent)
                cv2.imwrite(save_path.as_posix(), show_image_all)
                show_image_list_all = []

    def vis_response_map(self, feat_render, feat_real, data):
        output_dir = self.test_cfg.get('vis_dir')
        rendered_depths = data['rendered_depths']
        meta_infos = data['meta_infos']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        internel_k = data['internel_k']
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        per_img_patch_num = data['per_img_patch_num']
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        if data.get('gt_masks', None) is not None:
            gt_flow = filter_flow_by_mask(gt_flow, data['gt_masks'], self.max_flow)
        valid_mask = torch.sum(gt_flow**2, dim=1).sqrt() < self.max_flow
        valid_mask = valid_mask[:, None].to(torch.float32)
        scale = 8
        downsampled_valid_mask = F.interpolate(
            valid_mask, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True)[:, 0, ...]
        downsampled_gt_flow = 1/scale * F.interpolate(
                gt_flow, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True) 
        real_images, rendered_images = data['real_images'], data['rendered_images']
        N, C, H, W = feat_real.size()
        corr = torch.matmul(feat_render.view(N, C, -1).permute(0, 2, 1), feat_real.view(N, C, -1)).view(N, H, W, H, W)
        batch_size = len(real_images)
        real_images_cv2 = tensor_image_to_cv2(real_images)
        rendered_images_cv2 = tensor_image_to_cv2(rendered_images)
        image_index = 0
        show_image_list_all = []
        for i in range(batch_size):
            meta_info = meta_infos[image_index]
            sequence = str(Path(meta_info['img_path']).parents[1].name)
            valid_points_y, valid_points_x = (downsampled_valid_mask[i] > 0.9).nonzero(as_tuple=True)
            num_valid_points = len(valid_points_x)
            if num_valid_points <= 10:
                continue

            point_x, point_y = 16, 16
            # index = 10
            # point_x, point_y = valid_points_x[index], valid_points_y[index]
            render_bbox = torch.tensor([[point_x*8-4, point_y*8-4, point_x*8+4, point_y*8+4]]).cpu().data.numpy()
            show_render_image = mmcv.imshow_bboxes(rendered_images_cv2[i], render_bbox, thickness=1, show=False)

            real_point_x, real_point_y = downsampled_gt_flow[i, 0, point_y, point_x] + point_x , downsampled_gt_flow[i, 1, point_y, point_x] + point_y
            real_bbox = torch.tensor([[real_point_x*8-4, real_point_y*8-4, real_point_x*8+4, real_point_y*8+4]]).cpu().data.numpy()
            show_real_image = mmcv.imshow_bboxes(real_images_cv2[i], real_bbox, thickness=1, show=False)
            show_image_list = [show_real_image, show_render_image]
            similarity = corr[i, point_y, point_x]
            similarity = similarity / similarity.max()
            similarity = F.interpolate(similarity[None, None], scale_factor=(scale, scale), mode='bilinear', align_corners=True)[0, 0]
            similarity = (similarity * 255).clamp(0, 255).byte().cpu().numpy()
            save_similarity = cv2.applyColorMap(similarity, cv2.COLORMAP_JET)
            # fuse heatmap and real image
            show_real_image = (0.6* save_similarity + 0.4*show_real_image).astype(np.uint8)
            show_image_list = [show_real_image, show_render_image]
            show_image = np.concatenate(show_image_list, axis=1)
            show_image_list_all.append(show_image)
            if i >= sum(per_img_patch_num[:image_index+1])-1:
                image_index += 1
                show_image_all = np.concatenate(show_image_list_all, axis=0)
                save_path = Path(output_dir).joinpath(sequence).joinpath(str(Path(meta_info['img_path']).stem) + "_response.png")
                mmcv.mkdir_or_exist(Path(save_path).parent)
                cv2.imwrite(save_path.as_posix(), show_image_all)
                show_image_list_all = []

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