import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import sample_pts
from .transformer import SparseToDenseTransformer, GeometricStructureEmbedding
from pointnet2_ops.pointnet2_utils import QueryAndGroup
from .pytorch_utils import SharedMLP, Conv1d


class GeoPointMatching(nn.Module):
    def __init__(self, hidden_dim=64, coarse_npoint=196, nblock=1, focusing_factor=3, return_feat=True):
        super(GeoPointMatching, self).__init__()
        # self.cfg = cfg
        self.input_dim = hidden_dim
        self.out_dim = hidden_dim
        self.coarse_npoint = coarse_npoint
        self.geo_embedding = GeometricStructureEmbedding(hidden_dim=hidden_dim, sigma_d=0.2, sigma_a=15, angle_k=3, reduction_a='max')

        self.nblock = nblock
        self.in_proj = nn.Linear(self.input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.out_dim)
        self.bg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * .02)
        self.return_feat = return_feat

        self.transformers = []
        for _ in range(self.nblock):
            self.transformers.append(SparseToDenseTransformer(
                hidden_dim,
                num_heads=4,
                sparse_blocks=['self', 'cross'],
                dropout=None,
                activation_fn='ReLU',
                focusing_factor=focusing_factor,
                with_bg_token=True,
                replace_bg_token=True
            ))
        self.transformers = nn.ModuleList(self.transformers)

    def forward(self, f1, f2, end_points):
        B = end_points['dense_po'].size(0)
        dense_po = end_points['dense_po'] # .clone()
        dense_pm = end_points['dense_pm'] # .clone()
        bg_point = torch.ones(dense_pm.size(0),1,3).float().to(dense_pm.device) * 100

        sparse_pm, fps_idx_m = sample_pts(
            dense_pm, self.coarse_npoint, return_index=True
        )
        sparse_po, fps_idx_o = sample_pts(
            dense_po, self.coarse_npoint, return_index=True
        )
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))

        
        f1_atten = torch.cat([self.bg_token.repeat(B,1,1), f1], dim=1) # adding bg
        f2_atten = torch.cat([self.bg_token.repeat(B,1,1), f2], dim=1) # adding bg

        # atten_list = []
        for idx in range(self.nblock):
            f1_atten, f2_atten = self.transformers[idx](f1_atten, geo_embedding_m, fps_idx_m, f2_atten, geo_embedding_o, fps_idx_o)

        if self.return_feat:
            return end_points, self.out_proj(f1_atten[:, 1:, :]), self.out_proj(f2_atten[:, 1:, :])
        else:
            return end_points


#------------------------------- origin PositionalEncoding ------------------------------#
class PositionalEncodingOrig(nn.Module):
    def __init__(self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True):
        super(PositionalEncodingOrig, self).__init__()
        self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
        self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz)
        input_dim = 6 if use_xyz else 3

        self.mlp1 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp2 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp3 = Conv1d(256, out_dim, 1, activation=None, bn=None)

    def forward(self, pts1, pts2=None):
        if pts2 is None:
            pts2 = pts1

        # scale1
        feat1 = self.group1(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        feat1 = self.mlp1(feat1)
        feat1 = F.max_pool2d(
            feat1, kernel_size=[1, feat1.size(3)]
        )

        # scale2
        feat2 = self.group2(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )                       # B, 6, 2048, 64
        feat2 = self.mlp2(feat2)    # B, 128, 2048, 64
        feat2 = F.max_pool2d(
            feat2, kernel_size=[1, feat2.size(3)]
        )                           # B, 128, 2048, 1

        feat = torch.cat([feat1, feat2], dim=1).squeeze(-1) # B, 256, 2048
        feat = self.mlp3(feat).transpose(1,2)               # B, 2048, 128
        return feat

def grouped_point_clouds_sample(point_clouds, grid_x, grid_y):
    B, C, H, W = point_clouds.size()
    point_clouds = point_clouds.view(B, C, 256, 256, W)
    point_clouds = point_clouds[:, :, grid_x, grid_y, :]
    point_clouds = point_clouds.view(B, C, -1, W)
    return point_clouds.contiguous()     # (B, 6, 1024, 32)

def point_clouds_sample(point_clouds, grid_x, grid_y):
    B, N, C = point_clouds.size()   # B, N, 3
    H = W = int(N ** 0.5)
    point_clouds = point_clouds.view(B, H, W, C)
    point_clouds = point_clouds[:, grid_x, grid_y, :]
    point_clouds = point_clouds.view(B, -1, C)
    return point_clouds.contiguous()     # (B, 6, 1024, 32)

#----------------------------------- changed PositionalEncoding --------------------------------#  
class PositionalSampleEncoding(nn.Module):
    def __init__(self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True):
        super(PositionalSampleEncoding, self).__init__()
        self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
        self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz)
        input_dim = 6 if use_xyz else 3

        self.mlp1 = SharedMLP([input_dim, 16, 32], bn=bn)
        self.mlp2 = SharedMLP([input_dim, 16, 32], bn=bn)
        self.mlp3 = Conv1d(64, out_dim, 1, activation=None, bn=None)

        # self.mlp1 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        # self.mlp2 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        # self.mlp3 = Conv1d(256, out_dim, 1, activation=None, bn=None)

    def forward(self, pts1, pts2=None):

        grid_x, grid_y = torch.meshgrid(torch.arange(4, 256, 8), torch.arange(4, 256, 8), indexing='ij')
        grid_x, grid_y = grid_x.to(pts1.device), grid_y.to(pts1.device)

        if pts2 is None:
            pts2 = point_clouds_sample(pts1, grid_x, grid_y)
        # pts2_sample = point_clouds_sample(pts2, grid_x, grid_y)
        # feat1_sample = self.group1(
        #         pts1.contiguous(), pts2_sample.contiguous(), pts1.transpose(1,2).contiguous()
        #     )

        # scale1
        feat1 = self.group1(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        # feat1 = grouped_point_clouds_sample(feat1, grid_x, grid_y)
        feat1 = self.mlp1(feat1)
        feat1 = F.max_pool2d(
            feat1, kernel_size=[1, feat1.size(3)]
        )

        # scale2
        feat2 = self.group2(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )                       # B, 6, 2048, 64
        # feat2 = grouped_point_clouds_sample(feat2, grid_x, grid_y)
        feat2 = self.mlp2(feat2)    # B, 128, 2048, 64
        feat2 = F.max_pool2d(
            feat2, kernel_size=[1, feat2.size(3)]
        )                           # B, 128, 2048, 1

        feat = torch.cat([feat1, feat2], dim=1).squeeze(-1) # B, 256, 2048
        feat = self.mlp3(feat).transpose(1,2)               # B, 2048, 128
        return feat

#----------------------------------- changed PositionalEncoding --------------------------------#  
class PositionalEncoding(nn.Module):
    def __init__(self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True):
        super(PositionalEncoding, self).__init__()
        self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
        self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz)
        input_dim = 6 if use_xyz else 3

        self.mlp1 = SharedMLP([input_dim, 16, 32], bn=bn)
        self.mlp2 = SharedMLP([input_dim, 16, 32], bn=bn)
        self.mlp3 = Conv1d(64, out_dim, 1, activation=None, bn=None)

    def forward(self, pts1, pts2=None):
        if pts2 is None:
            pts2 = pts1

        # scale1
        feat1 = self.group1(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        feat1 = self.mlp1(feat1)
        feat1 = F.max_pool2d(
            feat1, kernel_size=[1, feat1.size(3)]
        )

        # scale2
        feat2 = self.group2(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )                       # B, 6, 2048, 64
        feat2 = self.mlp2(feat2)    # B, 128, 2048, 64
        feat2 = F.max_pool2d(
            feat2, kernel_size=[1, feat2.size(3)]
        )                           # B, 128, 2048, 1

        feat = torch.cat([feat1, feat2], dim=1).squeeze(-1) # B, 256, 2048
        feat = self.mlp3(feat).transpose(1,2)               # B, 2048, 128
        return feat
