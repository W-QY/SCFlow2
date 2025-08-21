from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .builder import REFINERS
from .base_refiner import BaseRefiner
from ..encoder import build_encoder
from ..loss import build_loss, RAFTLoss
from models.utils import (
    depth_to_point_cloud, 
    remap_pose_to_origin_resoluaion, 
)
from pointnet2_ops.pointnet2_utils import gather_operation, furthest_point_sample
from ..utils.bop_object_utils import load_objs, get_model_info
from ..utils.geo_point_matching import PositionalEncoding, PositionalSampleEncoding, PositionalEncodingOrig
from ..utils.dense_fusion import DenseFusion
from ..utils.transformer import feature_add_position, FeatureTransformer
from mmcv.cnn import build_conv_layer

@REFINERS.register_module()
class SCFlow2Refiner(BaseRefiner):
    def __init__(self,
                 seperate_encoder: bool,
                 cxt_channels: int,
                 h_channels: int,
                 cxt_encoder: dict,
                 encoder: dict,
                 decoder: dict,
                 renderer: dict,
                 cxt_feat_detach: bool = False,
                 max_flow: float = 400,
                 solve_type: str = 'reg',
                 add_dense_fusion: bool = True,
                 render_augmentations:list = None,
                 filter_invalid_flow: bool = True,
                 freeze_encoder: bool = False,
                 freeze_bn: bool = False,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(
            encoder=encoder, 
            decoder=decoder, 
            seperate_encoder=seperate_encoder, 
            renderer=renderer, 
            render_augmentations=render_augmentations,
            max_flow=max_flow,
            train_cfg=train_cfg, 
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        
        self.cxt_feat_detach = cxt_feat_detach
        if self.cxt_feat_detach:
            self.context = None
            self.context_out = build_conv_layer(
                None, self.render_encoder._dinov2_out_channels[cxt_encoder.net_type], cxt_encoder.out_channels, kernel_size=1)
        else:
            self.context = build_encoder(cxt_encoder)
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels


        assert self.h_channels == self.decoder.h_channels
        assert self.cxt_channels == self.decoder.cxt_channels
        self.freeze_all_bn = freeze_bn
        self.add_dense_fusion = add_dense_fusion
        if freeze_bn:
            self.freeze_bn()
        if freeze_encoder:
            self.freeze_encoder()
        self.filter_invalid_flow = filter_invalid_flow
        self.test_by_flow = self.test_cfg.get('by_flow', False)
        self.test_iter_num = self.test_cfg.get('iters') if 'iters' in self.test_cfg else self.decoder.iters
        self.solve_type = solve_type
        if self.add_dense_fusion:
            self.point_cloud_encoder = PositionalSampleEncoding(out_dim=64, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True)
            self.dense_fusion = DenseFusion(num_points=1024)
            self.cxt_point_cloud_encoder = PositionalSampleEncoding(out_dim=64, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True)
            self.cxt_dense_fusion = DenseFusion(num_points=1024, out_channel=self.h_channels + self.cxt_channels)
    def freeze_encoder(self):
        for m in self.real_encoder.modules():
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for m in self.render_encoder.modules():
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def extract_feat(
        self, render_images, real_images,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The feature from the first
                image, the feature from the second image, the hidden state
                feature for GRU cell and the contextual feature.
        """
        # extract rgb feat
        if self.real_encoder.__class__.__name__ == 'DINOv2Encoder':
            real_feat, _ = self.real_encoder(real_images)
            render_feat, patch_features = self.render_encoder(render_images)
            if self.cxt_feat_detach:
                cxt_feat = self.context_out(patch_features.detach())
            else:
                cxt_feat = self.context(render_images)
        else:
            real_feat = self.real_encoder(real_images)
            render_feat = self.render_encoder(render_images)
            cxt_feat = self.context(render_images)

        h_feat, cxt_feat = torch.split(
            cxt_feat, [self.h_channels, self.cxt_channels], dim=1)
        h_feat = torch.tanh(h_feat)
        cxt_feat = torch.relu(cxt_feat)

        return render_feat, real_feat, h_feat, cxt_feat

    def extract_feat_with_depth_fusion(
        self, render_images, real_images, rendered_depths, gt_depths, internel_k
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The feature from the first
                image, the feature from the second image, the hidden state
                feature for GRU cell and the contextual feature.
        """
        # extract rgb feat
        if self.real_encoder.__class__.__name__ == 'DINOv2Encoder':
            real_feat, _ = self.real_encoder(real_images)
            render_feat, patch_features = self.render_encoder(render_images)
            if self.cxt_feat_detach:
                cxt_feat = self.context_out(patch_features.detach())
            else:
                cxt_feat = self.context(render_images)
        else:
            real_feat = self.real_encoder(real_images)
            render_feat = self.render_encoder(render_images)
            cxt_feat = self.context(render_images)


        B, C_f, H_f, W_f = render_feat.size()
        cxt_feat_channel = 512
        point_clouds_gt, point_clouds_rd = depth_to_point_cloud(gt_depths.detach(), rendered_depths.detach(), internel_k)

        point_feature_gt = self.point_cloud_encoder(point_clouds_gt.detach()).transpose(1, 2)
        point_feature_rd = self.point_cloud_encoder(point_clouds_rd.detach()).transpose(1, 2)
        cxt_point_feature_rd = self.cxt_point_cloud_encoder(point_clouds_rd.detach()).transpose(1, 2)

        fused_feat_real = self.dense_fusion(point_feature_gt, real_feat.view(B, C_f, H_f * W_f)).view(B, C_f, H_f, W_f)
        fused_feat_render = self.dense_fusion(point_feature_rd, render_feat.view(B, C_f, H_f * W_f)).view(B, C_f, H_f, W_f)
        fused_cxt_feat = self.cxt_dense_fusion(cxt_point_feature_rd, cxt_feat.view(B, C_f, H_f * W_f)).view(B, cxt_feat_channel, H_f, W_f)
        
        fused_h_feat, fused_cxt_feat = torch.split(
            fused_cxt_feat, [self.h_channels, self.cxt_channels], dim=1)
        fused_h_feat = torch.tanh(fused_h_feat)
        fused_cxt_feat = torch.relu(fused_cxt_feat)
        # 256 256 128 384（ctx 384 for raft3d, 128 for raft）
        return fused_feat_render, fused_feat_real, fused_h_feat, fused_cxt_feat, point_feature_rd.view(B, -1, H_f, W_f), point_feature_gt.view(B, -1, H_f, W_f)

    def get_pose_from_flow_and_depth(
            self,
            render_images,
            real_images,
            ref_rotation,
            ref_translation,
            rendered_depths,
            internel_k,
            label,
            gt_depths=None,
            init_flow=None,
            end_points=None,
    ) -> Dict[str, torch.Tensor]:
        """Forward function for RAFT when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.

        Returns:
            Dict[str, Tensor]: The losses of output.
        """

        # feat_render, feat_real, h_feat, cxt_feat = self.extract_feat(render_images, real_images)
        if self.add_dense_fusion:
            feat_render, feat_real, h_feat, cxt_feat, point_feat_rd, point_feat_gt = self.extract_feat_with_depth_fusion(
                            render_images, real_images, rendered_depths, gt_depths, internel_k)
        else:
            feat_render, feat_real, h_feat, cxt_feat = self.extract_feat(render_images, real_images)
            point_feat_rd = point_feat_gt = None

        if init_flow is None:
            N, _, H, W = real_images.shape
            init_flow = feat_render.new_zeros((N, 2, H, W), dtype=torch.float32, device=feat_render.device)

        return self.decoder(
            feat_render=feat_render, feat_real=feat_real, h_feat=h_feat, cxt_feat=cxt_feat,
            point_feat_rd=point_feat_rd, point_feat_gt=point_feat_gt,
            ref_rotation=ref_rotation, ref_translation=ref_translation,
            rendered_depths=rendered_depths, internel_k=internel_k, 
            label=label, init_flow=init_flow, 
            invalid_flow_num=0., 
            gt_depths=gt_depths,
            points_dict=end_points,
            )
    
    def forward_fdpose(self, data):
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        internel_k, rendered_depths, gt_depths = data['internel_k'], data['rendered_depths'], data['real_depths'] 
        feat_render, feat_real, h_feat, cxt_feat, point_feat_rd, point_feat_gt = self.extract_feat_with_depth_fusion(
                        rendered_images, real_images, rendered_depths, gt_depths, internel_k)
        
        input, output = {}, {}
        label = torch.tensor([]).to(real_images.device)
        N, _, H, W = real_images.shape
        init_flow = feat_render.new_zeros((N, 2, H, W), dtype=torch.float32, device=feat_render.device)

        flow_from_pose, flow_from_pose0, flow_from_pred, rotation_preds, translation_preds, \
            mask_preds, geo_rotation_preds, geo_translation_preds, points_dict  = \
                self.decoder(feat_render=feat_render, feat_real=feat_real, h_feat=h_feat, cxt_feat=cxt_feat,
                            point_feat_rd=point_feat_rd, point_feat_gt=point_feat_gt,
                            ref_rotation=ref_rotations, ref_translation=ref_translations,
                            rendered_depths=rendered_depths, internel_k=internel_k, 
                            label=label, init_flow=init_flow, 
                            invalid_flow_num=0., 
                            gt_depths=gt_depths,
                            points_dict=input,
                            )
        output['rot'] = geo_rotation_preds[-1]
        output['trans'] = geo_translation_preds[-1] / 1000.0
        return output

    def forward_single_pass(self, data, data_batch, return_loss=False):
        labels = data['labels']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        internel_k, rendered_depths, rendered_masks = data['internel_k'], data['rendered_depths'], data['rendered_masks']
        per_img_patch_num = data['per_img_patch_num']
        gt_depths = data['real_depths']
        cloud_list = data['cloud_list']
        model_list = data['model_list']
        inputs = {}
        model_list_convert = torch.matmul(ref_rotations.detach(), model_list.detach().transpose(1, 2)).transpose(1, 2) + ref_translations.unsqueeze(1).detach() / 1000.0

        n_instance = rendered_images.size(0)
        batch_size = n_instance      # 8 # 240603
        n_batch = int(np.ceil(n_instance/batch_size))
        for j in range(n_batch):
            start_idx = j * batch_size
            end_idx = n_instance if j == n_batch-1 else (j+1) * batch_size
            inputs['dense_pm'] = cloud_list[start_idx:end_idx].contiguous()
            inputs['dense_po'] = model_list[start_idx:end_idx].contiguous()
            inputs['dense_po_convert'] = model_list_convert[start_idx:end_idx].contiguous()
            sequence_flow_from_pose2, sequence_flow_from_pose1, sequence_flow_from_pred, seq_rotations, seq_translations, sequence_masks, geo_rotation_preds, geo_translation_preds, end_points = \
                self.get_pose_from_flow_and_depth(
                    rendered_images, real_images,
                    ref_rotations, ref_translations, 
                    rendered_depths, internel_k, labels, gt_depths=gt_depths, end_points=inputs
                )

        solve_type = self.solve_type  # reg pnp kabsch
        if solve_type == 'reg':
            batch_rotations = geo_rotation_preds[-1]
            batch_translations = geo_translation_preds[-1]
            batch_rotations = torch.split(batch_rotations, per_img_patch_num)
            batch_translations = torch.split(batch_translations, per_img_patch_num)
            batch_labels = torch.split(labels, per_img_patch_num)
            batch_scores = torch.split(torch.ones_like(labels, dtype=torch.float32), per_img_patch_num)
        elif solve_type == 'pnp':
            results = self.solve_pose(
                sequence_flow_from_pred[-1], rendered_depths, ref_rotations, ref_translations, internel_k, labels, per_img_patch_num, sequence_masks[-1].squeeze(dim=1))
            batch_rotations, batch_translations = results['rotations'], results['translations']
            batch_labels, batch_scores = results['labels'], results['scores']
        elif solve_type == 'kabsch':
            results = self.solve_pose_depth(
                sequence_flow_from_pred[-1], rendered_depths, gt_depths, ref_rotations, ref_translations, \
                internel_k, labels, per_img_patch_num, sequence_masks[-1].squeeze(dim=1))
            batch_rotations, batch_translations = results['rotations'], results['translations']
            batch_labels, batch_scores = results['labels'], results['scores']

        image_metas = data_batch['img_metas']
        batch_internel_k = torch.split(internel_k, per_img_patch_num)

        if self.test_cfg.get('vis_result', False):
            results=dict(rotations=batch_rotations, translations=batch_translations, scores=batch_scores, labels=batch_labels) 
            # if self.test_cfg.get('vis_result', False):
            self.visualize_and_save(data, sequence_flow_from_pred, torch.cat(results['rotations']), torch.cat(results['translations']), sequence_flow_from_pose2)
            if self.test_cfg.get('vis_seq_flow', False):
                self.visualize_sequence_flow_and_fw(data, sequence_flow_from_pose2)

        batch_rotations, batch_translations = remap_pose_to_origin_resoluaion(batch_rotations, batch_translations, batch_internel_k, image_metas)
        return dict(
            rotations = batch_rotations,
            translations = batch_translations,
            labels = batch_labels,
            scores = batch_scores,
        )


