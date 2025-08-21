import kornia
import torch
from torch import nn
from .builder import LOSSES
from .utils import weighted_loss




def robust_l1(x):
    return (x**2 + 0.001**2)**0.5

@weighted_loss
def edge_aware_smoothness(image, flow, mode, order=2, edge_constant=100.):
    '''Compute Edge aware smoothness  https://arxiv.org/abs/2006.04902

    Args:
        image (torch.Tensor): shape (B,C,H,W)
        flow (torch.Tensor): shape (B,2,H,W)
        order (1|2): gradient order for flow, refer to k_th smoothness in the paper.
        edge_constant (float): lambda
    return:
        smoothness (torch.Tensor): shape (B,H,W)
    
    '''
    image_grads = kornia.filters.spatial_gradient(image, mode, order=order, normalized=False)

    image_grads_x, image_grads_y = image_grads[:, :, 0], image_grads[:, :, 1]
    weights_x = torch.exp(
        -torch.mean(torch.abs(edge_constant * image_grads_x), dim=1, keepdim=True)
    )
    weights_y = torch.exp(
        -torch.mean(torch.abs(edge_constant * image_grads_y), dim=1, keepdim=True)
    )

    # TODO Inconsisent with the origin implement
    flow_grads = kornia.filters.spatial_gradient(flow, mode, order=order, normalized=False)
    flow_grads_x, flow_grads_y = flow_grads[:, :, 0], flow_grads[:, :, 1]

    smoothness_x = torch.mean(weights_x * robust_l1(flow_grads_x), dim=1)
    smoothness_y = torch.mean(weights_y * robust_l1(flow_grads_y), dim=1)
    smoothness = (smoothness_x + smoothness_y) / 2
    return smoothness

@LOSSES.register_module()
class EdgeAwareSmoothness(nn.Module):
    def __init__(self,
                gradient_mode='sobel',
                gradient_order=1,
                edge_constant=100.,
                reduction='none',
                loss_weight=1.0,):
        super().__init__()
        self.gradient_order = gradient_order
        self.gradient_mode = gradient_mode
        self.edge_constant = edge_constant
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, 
                image,
                flow,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert image.shape[-2:] == flow.shape[-2:]

        loss = self.loss_weight * edge_aware_smoothness(
            image,
            flow,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            mode=self.gradient_mode,
            order=self.gradient_order,
            edge_constant=self.edge_constant
        )
        return loss



@LOSSES.register_module()
class ZeroSmoothness(nn.Module):
    def __init__(self,
                eps=0.001,
                p=0.5,
                reduction='none',
                loss_weight=1.0):
        super().__init__()
        self.reduction =  reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.p = p

    @weighted_loss
    def robust_l1(self, x):
        return (x**2 + self.eps**2)**self.p
    
    def forward(self, 
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * self.robust_l1(
            pred,
            reduction=reduction,
            weight=weight,
            avg_factor=avg_factor,
        )
        return loss
                