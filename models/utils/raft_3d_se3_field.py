import torch
import torch.nn.functional as F
from lietorch import SE3
import lietorch_extras

MIN_DEPTH = 0.02

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def depth_sampler(depths, coords):
    depths_proj, valid = bilinear_sampler(depths[:,None], coords, mask=True)
    return depths_proj.squeeze(dim=1), valid


def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:,None,None].unbind(dim=-1)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    # 1.0 / Z: defatult RAFT3D; Z: SCFlow2
    # d = 1.0 / Z   
    d = Z

    coords = torch.stack([x, y, d], dim=-1)
    return coords

def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape[-2:]
    
    fx, fy, cx, cy = \
        intrinsics[:,None,None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(), 
        torch.arange(wd).to(depths.device).float())

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return torch.stack([X, Y, Z], dim=-1)   # to point cloud

def projective_transform(Ts, depth, intrinsics):
    """ Project points from I1 to I2 """
    
    X0 = inv_project(depth, intrinsics)     # to point cloud
    X1 = Ts * X0
    x1 = project(X1, intrinsics)

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return x1, valid.float()

def induced_flow(Ts, depth, intrinsics):
    """ Compute 2d and 3d flow fields """

    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0

    x0 = project(X0, intrinsics)
    x1 = project(X1, intrinsics)

    flow2d = x1 - x0
    flow3d = X1 - X0

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return flow2d, flow3d, valid.float()


def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = \
        intrinsics[None].unbind(dim=-1)
    
    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(), 
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = torch.stack([X1-X0, Y1-Y0, Z1-Z0], dim=-1)
    return flow3d

class SE3BuilderInplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, se3, ae, pts, target, weight, intrinsics, radius=32):
        """ Build linear system Hx = b """
        ctx.radius = radius
        ctx.save_for_backward(se3, ae, pts, target, weight, intrinsics)
        
        H, b = lietorch_extras.se3_build_inplace(
            se3, ae, pts, target, weight, intrinsics, radius)
        
        return H, b

    @staticmethod
    def backward(ctx, grad_H, grad_b):
        se3, ae, pts, target, weight, intrinsics = ctx.saved_tensors
        ae_grad, target_grad, weight_grad = lietorch_extras.se3_build_inplace_backward(
            se3, ae, pts, target, weight, intrinsics, grad_H, grad_b, ctx.radius)

        return None, ae_grad, None, target_grad, weight_grad, None


class SE3Builder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn, se3, pts, target, weight, intrinsics, radius=32):
        """ Build linear system Hx = b """
        ctx.radius = radius
        ctx.save_for_backward(attn, se3, pts, target, weight, intrinsics)
        
        H, b = lietorch_extras.se3_build(
            attn, se3, pts, target, weight, intrinsics, radius)
        
        return H, b

    @staticmethod
    def backward(ctx, grad_H, grad_b):
        attn, se3, pts, target, weight, intrinsics = ctx.saved_tensors
        grad_H = grad_H.contiguous()
        grad_b = grad_b.contiguous()
        attn_grad, target_grad, weight_grad = lietorch_extras.se3_build_backward(
            attn, se3, pts, target, weight, intrinsics, grad_H, grad_b, ctx.radius)

        return attn_grad, None, None, target_grad, weight_grad, None


class SE3Solver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        ctx.save_for_backward(H, b)
        x, = lietorch_extras.cholesky6x6_forward(H, b)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        H, b = ctx.saved_tensors
        grad_x = grad_x.contiguous()
        
        grad_H, grad_b = lietorch_extras.cholesky6x6_backward(H, b, grad_x)
        return grad_H, grad_b


class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

def block_solve(H, b, ep=0.1, lm=0.0001):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    I = torch.eye(D).to(H.device)
    H = H + (ep + lm*H) * I

    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)
    b = b.reshape(B, N*D, 1)

    x = CholeskySolver.apply(H,b)
    return x.reshape(B, N, D)


def attention_matrix(X):
    """ compute similiarity matrix between all pairs of embeddings """
    batch, ch, ht, wd = X.shape
    X = X.view(batch, ch, ht*wd) / 8.0

    dist = -torch.sum(X**2, dim=1).view(batch, 1, ht*wd) + \
           -torch.sum(X**2, dim=1).view(batch, ht*wd, 1) + \
           2 * torch.matmul(X.transpose(1,2), X)

    A = torch.sigmoid(dist)
    return A.view(batch, ht, wd, ht, wd)


def step(Ts, ae, target, weight, depth, intrinsics, lm=.0001, ep=10.0):
    """ dense gauss newton update """
    
    pts = inv_project(depth, intrinsics)
    pts = pts.permute(0,3,1,2).contiguous()
    
    attn = attention_matrix(ae)
    se3 = Ts.matrix().permute(0,3,4,1,2).contiguous()

    # build the linear system
    H, b = SE3Builder.apply(attn, se3, pts, target, weight, intrinsics)

    I = torch.eye(6, device=H.device)[...,None,None]
    H = H + (lm*H + ep) * I  # damping

    dx = SE3Solver.apply(H, b)
    dx = dx.permute(0,3,4,1,2).squeeze(-1).contiguous()

    Ts = SE3.exp(dx) * Ts
    return Ts


def step_inplace(Ts, ae, target, weight, depth, intrinsics, lm=.0001, ep=10.0):
    """ dense gauss newton update with computing similiarity matrix """
    
    pts = inv_project(depth, intrinsics)   # depth to point cloud
    pts = pts.permute(0,3,1,2).contiguous()

    # tensor representation of SE3
    se3 = Ts.data.permute(0,3,1,2).contiguous()
    ae = ae / 8.0

    # build the linear system
    H, b = SE3BuilderInplace.apply(se3, ae, pts, target, weight, intrinsics)

    I = torch.eye(6, device=H.device)[...,None,None]
    H = H + (lm*H + ep) * I  # damping

    dx = SE3Solver.apply(H, b)
    dx = dx.permute(0,3,4,1,2).squeeze(-1).contiguous()

    Ts = SE3.exp(dx) * Ts
    return Ts

def cvx_upsample(data, mask):
    """ convex combination upsampling (see RAFT) """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)
    return up_data

def upsample_se3(Ts, mask):
    """ upsample a se3 field """
    tau_phi = Ts.log()
    return SE3.exp(cvx_upsample(tau_phi, mask))

def upsample_flow(flow, mask):
    """ upsample a flow field """
    flow = flow * torch.as_tensor([8.0, 8.0, 1.0]).to(flow.device)
    return cvx_upsample(flow, mask)
