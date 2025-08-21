import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
from sksparse import cholmod
from lietorch import SE3

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
    if depth2.ndimension() > 1:
        depth2_r8 = depth2[:,3::8,3::8] / 1000.0
        return Ts, ae, coords0, depth1_r8, depth2_r8, intrinsics_r8
    else:
        return Ts, ae, coords0, depth1_r8, None, intrinsics_r8

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


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, rgbd_version=True):
        super(BasicUpdateBlock, self).__init__()
        self.rgbd_version = rgbd_version
        self.gru = ConvGRU(hidden_dim, dilation=3)

        self.solver = GridSmoother()

        self.corr_enc = nn.Sequential(
            # nn.Conv2d(196, 256, 3, padding=1),
            nn.Conv2d(324, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*128, 1, padding=0))

        if self.rgbd_version:
            flow_enc_inputdim = 9
        else:
            flow_enc_inputdim = 8
        self.flow_enc = nn.Sequential(
            nn.Conv2d(flow_enc_inputdim, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*128, 1, padding=0))

        self.ae_enc = nn.Conv2d(16, 3*128, 3, padding=1)

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip(),
            nn.Sigmoid())

        self.ae_wts = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, padding=0),
            GradientClip(),
            nn.Softplus())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0),
            GradientClip())


    def forward(self, net, inp, corr, flow, twist, dz, ae, upsample=True):
        if self.rgbd_version:
            motion_info = torch.cat([flow, 10*twist, 10*dz], dim=-1)
        else:
            motion_info = torch.cat([flow, 10*twist], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0,3,1,2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        ae = self.ae_enc(ae)
        net = self.gru(net, inp, cor, mot, ae)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        edges = 5 * self.ae_wts(net)
        ae = self.solver(ae, edges)

        return net, mask, ae, delta, weight

GRAD_CLIP = .01
class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x


class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)

class GridSmoother(nn.Module):
    def __init__(self):
        super(GridSmoother, self).__init__()

    def forward(self, ae, wxwy):

        factor = GridFactor()
        ae = ae.permute(0,2,3,1)

        wx = wxwy[:,0].unsqueeze(-1)
        wy = wxwy[:,1].unsqueeze(-1)
        wu = torch.ones_like(wx)
        J = torch.ones_like(wu).unsqueeze(-2)

        # residual terms
        ru = ae.unsqueeze(-2)
        rx = torch.zeros_like(ru)
        ry = torch.zeros_like(ru)

        factor.add_factor([J], wu, ru, ftype='u')
        factor.add_factor([J,-J], wx, rx, ftype='h')
        factor.add_factor([J,-J], wy, ry, ftype='v')
        factor._build_factors()

        ae = factor.solveAAt().squeeze(dim=-2)
        ae = ae.permute(0,3,1,2).contiguous()

        return ae


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128, dilation=4):
        super(ConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.convz1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convz2 = nn.Conv2d(hidden_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

        self.convr1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convr2 = nn.Conv2d(hidden_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

        self.convq1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convq2 = nn.Conv2d(hidden_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

    def forward(self, h, *inputs):
        iz, ir, iq = 0, 0, 0
        for inp in inputs:
            inp = inp.split([self.hidden_dim]*3, dim=1)
            iz = iz + inp[0]
            ir = ir + inp[1]
            iq = iq + inp[2]

        z = torch.sigmoid(self.convz1(h) + self.convz2(h) + iz)
        r = torch.sigmoid(self.convr1(h) + self.convr2(h) + ir)
        q = torch.tanh(self.convq1(r*h) + self.convq2(r*h) + iq)

        h = (1-z) * h + z * q
        return h


class GridCholeskySolver(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, chols, J, w, b):
        """ Solve linear system """
        B, H, W, M, N = J.shape
        D = b.shape[-1]
        bs = b.detach().reshape(B, -1, D).cpu().numpy()
        
        xs = []
        for i in range(len(chols)):
            xs += [ chols[i](bs[i]) ]

        xs = np.stack(xs).astype(np.float32)
        xs = torch.from_numpy(xs).to(J.device)
        xs = xs.view(B, H, W, N//4, D)

        ctx.chols = chols
        ctx.save_for_backward(xs, J, w, b)
        return xs

    @staticmethod
    def backward(ctx, grad_output):
        xs, J, w, b = ctx.saved_tensors
        B, H, W, M, N = J.shape
        D = b.shape[-1]

        gs = grad_output.reshape(B, -1, D).cpu().numpy()
        chols = ctx.chols
        
        dz = []
        for i in range(len(chols)):
            dz += [ chols[i](gs[i]) ]

        dz = np.stack(dz, axis=0).astype(np.float32)
        dz = torch.from_numpy(dz).to(J.device).view(*xs.shape)

        J = GridFactor(A=J, w=w)

        grad_J = torch.matmul(-w[...,None] * J.A(dz), J._unfold(xs).transpose(-1,-2)) + \
                 torch.matmul(-w[...,None] * J.A(xs), J._unfold(dz).transpose(-1,-2))
        
        grad_w = -torch.sum(J.A(xs) * J.A(dz), -1)

        return None, grad_J, grad_w, dz


sym_factor = None
sym_shape = None
class GridFactor:
    """ Generalized grid factors """
    def __init__(self, A=None, w=None):
        self.factors = []
        self.weights = []
        self.residuals = []
        
        self.chols = None
        self.Af = A
        self.wf = w

    def _build_factors(self):
        self.Af = torch.cat(self.factors, dim=3)
        self.wf = torch.cat(self.weights, dim=3)

    def add_factor(self, Js, ws=None, rs=None, ftype='u'):
        """ Add factor to graph """

        B, H, W, M, N = Js[0].shape
        device = Js[0].device

        A = torch.zeros([B, H, W, M, N, 2, 2]).to(device)
        w = torch.zeros([B, H, W, M]).to(device)
        
        # unary factor
        if ftype == 'u':
            A[...,0,0] = Js[0]
            w[:] = ws[:]
        
        # horizontal pairwise factor
        elif ftype == 'h':
            A[...,0,0] = Js[0]
            A[...,0,1] = Js[1]
            w[:, :, :-1, :] = ws[:, :, :-1, :]

        # verticle pairwise factor 
        elif ftype == 'v':
            A[...,0,0] = Js[0]
            A[...,1,0] = Js[1]
            w[:, :-1, :, :] = ws[:, :-1, :, :]

        A = A.view(B, H, W, M, 2*2*N)

        self.factors.append(A)
        self.weights.append(w)

        if rs is not None:
            self.residuals.append(rs)

    def _fold(self, x):
        """ Transposed fold operator """
        B, H, W, M, D = x.shape
        x = x.transpose(-1,-2)
        x = x.reshape(B, H, W, M*D)
        x = F.pad(x, [0,0,1,0,1,0])
        x = x.reshape(B, (H+1)*(W+1), M*D).permute(0, 2, 1)
        x = F.fold(x, [H, W], [2,2], padding=1)
        x = x.permute(0, 2, 3, 1).reshape(B, H, W, D, M//4)
        return x.transpose(-1,-2)

    def _unfold(self, x):
        """ Transposed unfold operator """
        B, H, W, N, D = x.shape
        x = x.transpose(-1,-2)
        x = F.pad(x.view(B, H, W, N*D), [0,0,0,1,0,1])
        x = x.permute(0, 3, 1, 2)
        x = F.unfold(x, [2,2], padding=0)
        x = x.permute(0, 2, 1).reshape(B, H, W, D, 4*N)
        return x.transpose(-1, -2)

    def A(self, x, w=False):
        """ Linear operator """
        return torch.matmul(self.Af, self._unfold(x))

    def At(self, y):
        """ Adjoint operator """
        w = self.wf.unsqueeze(dim=-1)
        At = self.Af.transpose(-1,-2)
        return self._fold(torch.matmul(At, w*y))

    def to_csc(self):
        """ Convert linear operator into scipy csc matrix"""

        if self.Af is None:
            self._build_factors()

        with torch.no_grad():
            B, H, W, N, M = self.Af.shape
            dims = [torch.arange(d).cuda() for d in (H, W, N, M//4)]

            i0, j0, k0, h0 = \
                [x.reshape(-1) for x in torch.meshgrid(*dims)]

            # repeats are ok because edge weights get zeroed
            s = [W*(M//4), M//4, 1]
            i1 = i0+1
            j1 = j0+1
            i1[i1 >= H] = H-1
            j1[j1 >= W] = W-1

            col_idx = torch.stack([
                s[0]*i0 + s[1]*j0 + s[2]*h0,
                s[0]*i0 + s[1]*j1 + s[2]*h0,
                s[0]*i1 + s[1]*j0 + s[2]*h0,
                s[0]*i1 + s[1]*j1 + s[2]*h0
            ], dim=-1).view(-1)

            dense_shape = [H*W*N, H*W*(M//4)]
            col_idx = col_idx.cpu().numpy()
            row_idx = M * np.arange(0, H*W*N+1)

            A = self.Af.detach().view(B, H*W*N, M)
            wsqrt = self.wf.detach().sqrt().view(B, H*W*N, 1)
            vals = (wsqrt * A).cpu().numpy()

            sparse_matricies = []
            for batch_ix in range(B):
                data = (vals[batch_ix].reshape(-1), col_idx, row_idx)
                mat = scipy.sparse.csr_matrix(data, shape=dense_shape)
                mat.sum_duplicates()
                sparse_matricies.append(mat.T)

        return sparse_matricies

    def factorAAt(self):
        """ Peform sparse cholesky factorization """
        global sym_factor, sym_shape

        with torch.no_grad():
            self.chols = []
            start = time.time()
            As = self.to_csc()

            if sym_factor is None or As[0].shape != sym_shape:
                sym_factor = cholmod.analyze_AAt(As[0], ordering_method='best')
                sym_shape = As[0].shape

            for A in As:
                chol = sym_factor.cholesky_AAt(A)
                self.chols.append(chol)

        return self.chols

    def solveAAt(self, b=None):
        if self.chols is None:
            self.factorAAt()

        if b is None:
            r = torch.cat(self.residuals, -2)
            b = self.At(r)

        x = GridCholeskySolver.apply(self.chols, self.Af, self.wf, b)
        return x.reshape(*b.shape)

