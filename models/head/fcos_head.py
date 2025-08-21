import torch
import torch.nn as nn
from torch.nn import functional as F
from .builder import HEAD
from mmcv.cnn import ConvModule, Scale, normal_init, bias_init_with_prob
from mmcv.runner import BaseModule



INF = 1000000
@HEAD.register_module()
class FCOS_HEAD(BaseModule):
    def __init__(self, 
                num_classes,
                in_channel,
                feat_channel,
                dcn_on_last_conv=False,
                keypoint_num=8,
                stacked_convs=4,
                num_levels=5,
                pointwise=False,
                conv_cfg=None,
                norm_cfg=dict(
                    type='GN', 
                    num_groups=32, 
                    requires_grad=True),
                init_cfg=dict(
                    type='Normal',
                    layer='Conv2d',
                    std=0.01)
                ):
        super(FCOS_HEAD, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.feat_channel = feat_channel
        self.stacked_convs = stacked_convs
        self.dcn_on_last_conv = dcn_on_last_conv
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.keypoint_num = keypoint_num
        if not pointwise:
            self.construct_layers()
        else:
            self.construct_layers(kernel_size=1, padding=0)
    
    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.pose_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_conv, std=0.01, bias=bias_cls)
        for keypoint_conv in self.keypoint_convs:
            normal_init(keypoint_conv, std=0.01, bias=1.)

    def construct_layers(self, kernel_size=3, padding=1):
        self.pose_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channel if i == 0 else self.feat_channel
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg 
            self.pose_convs.append(
                ConvModule(
                    chn,
                    self.feat_channel,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channel,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
        self.cls_conv = nn.Conv2d(self.feat_channel, self.num_classes, kernel_size, padding=padding)
        self.keypoint_convs = nn.ModuleList([nn.Conv2d(self.feat_channel, self.num_classes*2, kernel_size, padding=padding) for _ in range(self.keypoint_num)])
        self.pose_scales = nn.ModuleList([Scale(1.0) for _ in range(self.num_levels)])

    
    def forward(self, feats, label):
        '''
        Forward features of mulit scale from FPN
        return:
            keypoints_preds (list[Tensor]): predicted keypoint regression of multiple levels, 
                shape (n, keypoint_num*2, h, w)
        '''
        assert len(feats) == self.num_levels
        keypoints_pred_multilvl = []
        cls_pred_multilvl = []
        for feat, pose_scale in zip(feats, self.pose_scales):
            pose_feat, cls_feat = feat, feat
            for pose_layer in self.pose_convs:
                pose_feat = pose_layer(pose_feat)
            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_pred = self.cls_conv(cls_feat)
            keypoints_pred = []
            for keypoint_conv in self.keypoint_convs:
                keypoint_pred = keypoint_conv(pose_feat)
                N, _, H, W = keypoint_pred.shape
                keypoint_pred = keypoint_pred.reshape(N, self.num_classes, 2, H, W)
                keypoints_pred.append(keypoint_pred)
            keypoints_pred = torch.stack(keypoints_pred, dim=2)
            # (N, num_classes, keypoint_num, 2, H, W)
            N, _, _, _, H, W = keypoints_pred.shape
            keypoints_pred = keypoints_pred[torch.arange(N), label]
            keypoints_pred = pose_scale(keypoints_pred).float()
            keypoints_pred_multilvl.append(keypoints_pred)
            cls_pred_multilvl.append(cls_pred)
        return cls_pred_multilvl, keypoints_pred_multilvl



@HEAD.register_module()
class Sym_FCOS_HEAD(BaseModule):
    def __init__(self, 
                num_classes, 
                symmetry_labels,
                in_channel, 
                feat_channel, 
                dcn_on_last_conv=False, 
                keypoint_num=8, 
                stacked_convs=4, 
                num_levels=5, 
                conv_cfg=None, 
                norm_cfg=..., 
                init_cfg=...):
        super(Sym_FCOS_HEAD, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.feat_channel = feat_channel
        self.stacked_convs = stacked_convs
        self.dcn_on_last_conv = dcn_on_last_conv
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.keypoint_num = keypoint_num
        self.symmetry_labels = symmetry_labels 
    
    def construct_layers(self):
        self.pose_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channel if i == 0 else self.feat_channel
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg 
            self.pose_convs.append(
                ConvModule(
                    chn,
                    self.feat_channel,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channel,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
        self.cls_conv = nn.Conv2d(self.feat_channel, self.num_classes, 3, padding=1)
        keypoint_convs, domain_classify_convs = [], [], []
        for i in range(1, self.num_classes+1):
            if i in self.symmetry_labels:
                keypoint_conv_domain_1 = nn.Conv2d(self.feat_channel, self.keypoint_num*2, 3, padding=1)
                keypoint_conv_domain_2 = nn.Conv2d(self.feat_channel, self.keypoint_num*2, 3, padding=1)
                keypoint_convs.append(nn.ModuleList([keypoint_conv_domain_1, keypoint_conv_domain_2]))
                domain_classify_convs.append(nn.Conv2d(self.feat_channel, 1, 3, padding=1))
            else:
                keypoint_convs.append(nn.Conv2d(self.feat_channel, self.keypoint_num*2, 3, padding=1))
                domain_classify_convs.append(nn.Identity()) # place holder
        self.keypoint_convs = nn.ModuleList(keypoint_convs)
        self.dommain_classify_convs = nn.ModuleList(domain_classify_convs)
        self.pose_scales = nn.ModuleList([Scale(1.0) for _ in range(self.num_levels)])
    
    def forward_keypoints_training(self, labels, sigmas, feats):
        keypoint_conv_parameters = []
        for i, label in enumerate(labels):
            keypoint_conv = self.keypoint_convs[label]
            if isinstance(keypoint_conv, nn.ModuleList):
                keypoint_conv = keypoint_conv[sigmas[i]]
            keypoint_conv_parameters.append(keypoint_conv.parameters)
        
        keypoint_conv_parameters = torch.stack(keypoint_conv_parameters)
        N, C, H, W = feats.shape
        keypoint_conv_parameters = keypoint_conv_parameters.view(N, C, 3, 3)
        feats = feats.view(1, -1, H, W)
        keypoints = F.conv2d(feats, keypoint_conv_parameters, stride=1, padding=1, groups=N)
        keypoints = keypoints.view(N, self.keypoint_num*2, H, W)
        return keypoints
    
    def forward_keypoints_testing(self, labels, feats):
        pass


    def forward(self, feats, labels, sigmas):
        assert len(feats) == self.num_levels
        keypoints_pred_multilvl, cls_pred_multilvl = [], []
        for feat, pose_scale in zip(feats, self.pose_scales):
            pose_feat, cls_feat = feat, feat 
            for pose_layer in self.pose_convs:
                pose_feat = pose_layer(pose_feat)
            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_pred = self.cls_conv(cls_feat)
            keypoints_pred = self.forward_keypints(labels, sigmas, pose_feat)
            keypoints_pred = pose_scale(keypoints_pred).float()
            keypoints_pred_multilvl.append(keypoints_pred)
            cls_pred_multilvl.append(cls_pred)
        return cls_pred_multilvl, keypoints_pred_multilvl