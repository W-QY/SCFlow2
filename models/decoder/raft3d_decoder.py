import torch
import torch.nn as nn
from ..utils.raft_3d_basic_blocks import BasicUpdateBlock
from ..utils.raft_3d_se3_field import (SE3, depth_sampler, projective_transform, induced_flow,
                                      step_inplace, cvx_upsample, upsample_se3)


class RAFT3DPoseHead(nn.Module):
    def __init__(self):
        super(RAFT3DPoseHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)

        self.Linear1 = nn.Linear(2048, 512)
        self.Linear2 = nn.Linear(512, 256)
        self.rotation_pred = nn.Linear(256, 6)
        self.translation_pred = nn.Linear(256, 3)

    def init_weights(self):
        # zero translation
        nn.init.zeros_(self.translation_pred.weight)
        nn.init.zeros_(self.translation_pred.bias)
        # identity quarention
        nn.init.zeros_(self.rotation_pred.weight)
        with torch.no_grad():
                self.rotation_pred.bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, Ts):
        N, H, W, _  = Ts.data.size()
        x = Ts.matrix().reshape(N, H, W, 16).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten_op(x)
        x = self.Linear1(x)
        x = self.Linear2(x)
        pred_rotation_delta = self.rotation_pred(x)
        pred_translation_delta = self.translation_pred(x)

        return pred_rotation_delta, pred_translation_delta


def raft3d_initializer(depth1, depth2, cam_K):
    """ Initialize coords and transformation maps """

    batch_size, ht, wd = depth1.shape
    device = depth1.device

    y0, x0 = torch.meshgrid(torch.arange(ht//8), torch.arange(wd//8))
    coords0 = torch.stack([x0, y0], dim=-1).float()
    coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

    Ts = SE3.Identity(batch_size, ht//8, wd//8, device=device)
    ae = torch.zeros(batch_size, 16, ht//8, wd//8, device=device)

    # intrinsics and depth at 1/8 resolution
    intrinsics = torch.stack([cam_K[:, 0, 0], cam_K[:, 1, 1], cam_K[:, 0, 2], cam_K[:, 1, 2]], dim=-1)  # fx fy cx cy
    intrinsics_r8 = intrinsics / 8.0
    depth1[depth1 == -1.0] = 0.0
    depth1_r8 = depth1[:,3::8,3::8] / 1000.0
    depth2_r8 = depth2[:,3::8,3::8] / 1000.0

    return Ts, ae, coords0, depth1_r8, depth2_r8, intrinsics_r8

class RAFT3D_Decoder(nn.Module):
    def __init__(self):
        super(RAFT3D_Decoder, self).__init__()

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 3
        self.update_block = BasicUpdateBlock(hidden_dim=hdim)

    def forward(self, Ts, ae, coords0, pose_flow, depth1_r8, depth2_r8, corr, net, inp, intrinsics_r8):
        """ Estimate optical flow between pair of frames """

        # for itr in range(iters):
        Ts = Ts.detach()

        coords1_xyz, valid_xyz = projective_transform(Ts, depth1_r8, intrinsics_r8)
        coords1_xyz[valid_xyz==0.0] = 0.0

        coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1)
        zinv, valid_zinv = depth_sampler(depth2_r8, coords1)
        zinv[valid_zinv.squeeze(-1)==0.0] = 0.0

        flow = coords1 - coords0
        dz = zinv.unsqueeze(-1) - zinv_proj
        twist = Ts.log()

        net, mask, ae, delta, weight = \
            self.update_block(net, inp, corr, flow, twist, dz, ae)

        target = coords1_xyz.permute(0,3,1,2) + delta
        target = target.contiguous()

        # Gauss-Newton step
        # Ts = se3_field.step(Ts, ae, target, weight, depth1_r8, intrinsics_r8)
        Ts = step_inplace(Ts, ae, target, weight, depth1_r8, intrinsics_r8)         # 修改1205

        # if train_mode:
        flow2d_rev = target.permute(0,2,3,1)[...,:2] - coords0
        flow2d_rev = cvx_upsample(8 * flow2d_rev, mask)   # flow up sample by mask weight

        Ts_up = upsample_se3(Ts, mask)                    # Ts up sample by mask weight
        # flow2d_est, flow3d_est, valid = induced_flow(Ts_up, depth1, intrinsics)    # Ts induced flow
        # -------------
        # if train_mode:
        #   return flow_est_list, flow_rev_list
        return Ts, Ts_up, ae, net, flow2d_rev.permute(0, 3, 1, 2)





