from typing import Dict, Optional, Sequence, Union
import math, kornia, torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from .builder import DECODERS
from ..utils import get_flow_from_delta_pose_and_points, get_pose_from_delta_pose, CorrLookup, cal_3d_2d_corr, depth_to_point_cloud
from .raft_decoder import MotionEncoder, XHead, ConvGRU, CorrelationPyramid
from ..head import build_head
from .raft3d_decoder import raft3d_initializer, RAFT3D_Decoder, RAFT3DPoseHead
from lietorch import SE3

@DECODERS.register_module()
class SCFlow2Decoder(BaseModule):
    """The decoder of RAFT Net.

    The decoder of RAFT Net, which outputs list of upsampled flow estimation.

    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        num_levels (int): Number of levels used when calculating
            correlation tensor.
        radius (int): Radius used when calculating correlation tensor.
        iters (int): Total iteration number of iterative update of RAFTDecoder.
        corr_op_cfg (dict): Config dict of correlation operator.
            Default: dict(type='CorrLookup').
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        feat_channels (Sequence(int)): features channels of prediction module.
        mask_channels (int): Output channels of mask prediction layer.
            Default: 64.
        conv_cfg (dict, optional): Config dict of convolution layers in motion
            encoder. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in motion encoder.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in motion
            encoder. Default: None.
    """
    _h_channels = {'Basic': 128, 'Small': 96}
    # _cxt_channels = {'Basic': 128, 'Small': 64}         # scflow
    _cxt_channels = {'Basic': 384, 'Small': 64}         # scflow2

    def __init__(
        self,
        net_type: str,
        num_levels: int,
        radius: int,
        iters: int,
        detach_flow: bool,
        detach_mask: bool,
        detach_pose: bool,
        mask_flow: bool,
        mask_corr: bool,
        pose_head_cfg: dict,
        depth_transform: str='exp',
        cxt_channels: int=384, 
        depth_based_upsample: bool=False,
        detach_depth_for_xy: bool=False,
        corr_lookup_cfg: dict = dict(align_corners=True),
        gru_type: str = 'SeqConv',
        feat_channels: Union[int, Sequence[int]] = 256,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert net_type in ['Basic', 'Small']
        assert type(feat_channels) in (int, tuple, list)
        self.corr_block = CorrelationPyramid(num_levels=num_levels)

        feat_channels = feat_channels if isinstance(tuple,
                                                    list) else [feat_channels]
        self.net_type = net_type
        self.num_levels = num_levels
        self.radius = radius
        self.detach_flow = detach_flow
        self.detach_mask = detach_mask
        self.detach_pose = detach_pose
        self.detach_depth_for_xy = detach_depth_for_xy
        self.mask_flow = mask_flow
        self.mask_corr = mask_corr
        self.depth_transform = depth_transform
        self.h_channels = self._h_channels.get(net_type)
        self.cxt_channels = cxt_channels
        self.iters = iters
        corr_lookup_cfg['radius'] = radius
        self.corr_lookup = CorrLookup(**corr_lookup_cfg)
        self.encoder = MotionEncoder(
            num_levels=num_levels,
            radius=radius,
            net_type=net_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gru_type = gru_type
        self.gru = self.make_gru_block()
        self.pose_pred = build_head(pose_head_cfg)
        self.flow_pred = XHead(self.h_channels, feat_channels, 2, x='flow')
        self.mask_pred = XHead(self.h_channels, feat_channels, 1, x='mask')
        self.delta_flow_encoder = nn.Sequential(*self.make_delta_flow_encoder(
            2, channels=[128, 64], kernels=[7, 3], paddings=[3, 1], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.mask_encoder = nn.Sequential(*self.make_delta_flow_encoder(
            1, channels=[64, 32], kernels=[3, 3], paddings=[1, 1], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        
        self.depth_based_upsample = depth_based_upsample
        if self.depth_based_upsample:
            self.weight_channels = 576
            self.upsample_weight_encoder = nn.Sequential(*self.make_delta_flow_encoder(
                256, channels=[256, 128], kernels=[3, 3], paddings=[1, 1], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.upsample_weight_pred = XHead(self.h_channels, feat_channels, self.weight_channels, x='mask')
        self.raft3d_decoder = RAFT3D_Decoder()
        self.raft3d_head = RAFT3DPoseHead()

    def upsample_flow(self,
                  flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex
        combination.

        Args:
            flow (Tensor): The optical flow with the shape [N, 2, H/8, W/8].
            mask (Tensor, optional): The leanable mask with shape
                [N, grid_size x scale x scale, H/8, H/8].

        Returns:
            Tensor: The output optical flow with the shape [N, 2, H, W].
        """
        scale = 2**(self.num_levels - 1)
        grid_size = self.radius * 2 + 1
        grid_side = int(math.sqrt(grid_size))
        N, _, H, W = flow.shape
        if mask is None:
            new_size = (scale * H, scale * W)
            return scale * F.interpolate(
                flow, size=new_size, mode='bilinear', align_corners=True)
        # predict a (Nx8×8×9xHxW) mask
        mask = mask.view(N, 1, grid_size, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        # extract local grid with 3x3 side  padding = grid_side//2
        upflow = F.unfold(scale * flow, [grid_side, grid_side], padding=1)
        # upflow with shape N, 2, 9, 1, 1, H, W
        upflow = upflow.view(N, 2, grid_size, 1, 1, H, W)

        # take a weighted combination over the neighborhood grid 3x3
        # upflow with shape N, 2, 8, 8, H, W
        upflow = torch.sum(mask * upflow, dim=2)
        upflow = upflow.permute(0, 1, 4, 2, 5, 3)
        return upflow.reshape(N, 2, scale * H, scale * W)
    
    def upsample_mask(self, 
                    occlusion: torch.Tensor,
                    mask:Optional[torch.Tensor] = None)->torch.Tensor:
        scale = 2**(self.num_levels - 1)
        grid_size = self.radius * 2 + 1
        grid_side = int(math.sqrt(grid_size))
        N, _, H, W = occlusion.shape
        if mask is None:
            new_size = (scale * H, scale * W)
            return F.interpolate(
                occlusion, size=new_size, mode='bilinear', align_corners=True)
        
        mask = mask.view(N, 1, grid_size, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        upocclusion = F.unfold(occlusion, [grid_side, grid_side], padding=1)
        upocclusion = upocclusion.view(N, 1, grid_size, 1, 1, H, W)
        upocclusion = torch.sum(upocclusion * mask, dim=2)
        upocclusion = upocclusion.permute(0, 1, 4, 2, 5, 3)
        return upocclusion.reshape(N, 1, scale*H, scale*W)

    def make_delta_flow_encoder(self, in_channel, channels, kernels, paddings, conv_cfg, norm_cfg, act_cfg):
        encoder = []

        for ch, k, p in zip(channels, kernels, paddings):

            encoder.append(
                ConvModule(
                    in_channels=in_channel,
                    out_channels=ch,
                    kernel_size=k,
                    padding=p,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channel = ch
        return encoder



    def make_gru_block(self):
        return ConvGRU(
            self.h_channels,
            self.encoder.out_channels[0] + 2 + self.cxt_channels,
            net_type=self.gru_type)
    
        
    def _downsample(self, flow:torch.Tensor, mask:torch.Tensor):
        scale = 2**(self.num_levels - 1)
        N, _, H, W = flow.shape

        mask = mask.view(N, 1, scale*scale, H/scale, W/scale)
        mask = torch.softmax(mask, dim=2)
        
        downflow = F.unfold(flow/scale, [scale, scale], padding=1, stride=scale)
        downflow = downflow.view(N, 2, scale*scale, H/scale, W/scale)
        # shape (N, 2, H/scale, W/scale)
        downflow = torch.sum(mask * downflow, dim=2)
        return downflow

    def forward(self,
                feat_render: torch.Tensor=None, feat_real: torch.Tensor=None,
                h_feat: torch.Tensor=None, cxt_feat: torch.Tensor=None,
                point_feat_rd: torch.Tensor=None, point_feat_gt: torch.Tensor=None, 
                ref_rotation: torch.Tensor=None, ref_translation: torch.Tensor=None,
                rendered_depths: torch.Tensor=None, internel_k: torch.Tensor=None, 
                label:torch.Tensor=None, 
                init_flow:torch.Tensor=None,
                invalid_flow_num: float=None, 
                gt_depths: torch.Tensor=None,
                with_svd: bool=False,
                points_dict: dict=None,
                ) -> Sequence[torch.Tensor]:
        """Forward function for RAFTDecoder.

        Args:
            feat1 (Tensor): The feature from the first input image, shape (N, C, H, W)
            feat2 (Tensor): The feature from the second input image, shape (N, C, H, W).
            h_feat (Tensor): The hidden state for GRU cell, shape (N, C, H, W).
            cxt_feat (Tensor): The contextual feature from the first image, shape (N, C, H, W).
            ref_rotation (Tensor): The rotation which is used to render the renderering image.
            ref_translation (Tensor): The translation which is used to render the rendering image.
            depth (Tensor): The depth for rendering images.
            internel_k (Tensor): The camera parameters.
            label (Tensor): The label for training.

        Returns:
            Sequence[Tensor]: The list of predicted optical flow.
        """
        N, H, W = rendered_depths.size()
        B, C_f, H_f, W_f = feat_render.size()

        invalid_flow_num=0.
        scale = 2**(self.num_levels - 1)
        update_rotation, update_translation = ref_rotation, ref_translation
        geo_update_rotation, geo_update_translation = ref_rotation, ref_translation
        points_2d_list, points_3d_list = [], []
        delta_rotation_preds, delta_translation_preds = [], []
        rotation_preds, translation_preds = [], []
        flow_from_pose, flow_from_pred, flow_from_pose0 = [], [], []
        mask_preds = []
        geo_rotation_preds, geo_translation_preds = [], []
        Ts_list = []
        for i in range(N):
            points_2d, points_3d = cal_3d_2d_corr(rendered_depths[i], internel_k[i], ref_rotation[i], ref_translation[i])
            points_2d_list.append(points_2d)
            points_3d_list.append(points_3d)

        flow = init_flow
        init_mask = torch.ones((N, 1, H, W), dtype=init_flow.dtype, device=init_flow.device)
        init_mask = F.interpolate(init_mask, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True)
        mask = init_mask
        Ts, ae, coords0, depth1_r8, depth2_r8, intrinsics_r8 = \
                            raft3d_initializer(rendered_depths.detach(), gt_depths.detach(), internel_k.detach())


        corr_pyramid = self.corr_block(feat_render, feat_real)

        for i in range(self.iters):
            if self.detach_flow:
                flow = flow.detach()
            if self.detach_mask:
                mask = mask.detach()
            flow = 1/scale * F.interpolate(
                flow, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True)
            corr = self.corr_lookup(corr_pyramid, flow)
            # mask occluded pixels for correlation
            if self.mask_corr:
                corr = corr * mask

            # ------------------------------------ raft 3d decoder -----------------------------------
            mask = self.mask_pred(h_feat)
            mask = torch.sigmoid(mask)
            Ts, Ts_up, ae, h_feat, flow_pred = \
                self.raft3d_decoder(Ts, ae, coords0, flow, depth1_r8, depth2_r8, corr, h_feat, cxt_feat, intrinsics_r8)
            delta_rotation, delta_translation = self.pose_pred(Ts)  # self.raft3d_head(Ts)

            upsample_mask_pred = F.interpolate(
                    mask, scale_factor=(scale, scale), mode='bilinear', align_corners=True)
            Ts_list.append(Ts_up)


            # compute updated pose
            update_rotation, update_translation = get_pose_from_delta_pose(
                delta_rotation, delta_translation, 
                geo_update_rotation.detach() if self.detach_pose else geo_update_rotation,
                geo_update_translation.detach() if self.detach_pose else geo_update_translation,
                depth_transform=self.depth_transform, 
                detach_depth_for_xy=self.detach_depth_for_xy
            )
            geo_update_rotation, geo_update_translation = update_rotation.detach(), update_translation.detach()
            geo_rotation_preds.append(geo_update_rotation)
            geo_translation_preds.append(geo_update_translation)
            flow = get_flow_from_delta_pose_and_points(
                update_rotation, update_translation, internel_k, 
                points_2d_list, points_3d_list, H, W, 
                invalid_num=invalid_flow_num
            )

            delta_R_flow = torch.matmul(geo_update_rotation, ref_rotation.transpose(1, 2))
            delta_T_flow = geo_update_translation - torch.bmm(delta_R_flow, ref_translation.unsqueeze(-1)).squeeze(-1)
            delta_R_quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(delta_R_flow)
            Ts = SE3(torch.cat((delta_T_flow.detach()/1000.0, delta_R_quaternion.detach()), dim=1).unsqueeze(1).unsqueeze(2).expand(N, H_f, W_f, 7).contiguous())

            rotation_preds.append(update_rotation)
            translation_preds.append(update_translation)
            delta_rotation_preds.append(delta_rotation)
            delta_translation_preds.append(delta_translation)
            flow_from_pose.append(flow)
            flow_from_pred.append(flow_pred)
            mask_preds.append(upsample_mask_pred)
            flow_from_pose0 = None
        if not self.training:
            return flow_from_pose, flow_from_pose0, flow_from_pred, rotation_preds, translation_preds, mask_preds, geo_rotation_preds, geo_translation_preds, points_dict
        else:
            return flow_from_pose, flow_from_pred, rotation_preds, translation_preds, mask_preds, geo_rotation_preds, geo_translation_preds, points_dict
    
