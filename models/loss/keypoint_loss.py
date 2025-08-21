import torch
from torch import nn
from .utils import weight_reduce_loss, weighted_loss
from .builder import LOSSES



def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss

@LOSSES.register_module()
class ImageSpaceLoss(nn.Module):
    def __init__(self, 
                reg_decoded_keypoints=True,
                image_resolution=256, 
                loss_weight=1.0, 
                reduction='mean', 
                loss_type='l1', 
                beta=2):
        super().__init__()
        self.reg_decoded_keypoints=reg_decoded_keypoints
        self.image_resolution = image_resolution
        self.reduction = reduction
        self.loss_weight = loss_weight
        assert loss_type in ['smooth_l1', 'l1', 'l2']
        if loss_type == 'smooth_l1':
            assert isinstance(beta, float)
            self.beta = beta 
        self.loss_type = loss_type
        
    
    def forward(self,
                pred,
                target_2d,
                coder, 
                anchors,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        '''
        Args:
            pred (tensor): shape (n, keypoint_num*2), decoded preds
            target_2d (tensor): shape (n, keypoint_num*2)
            label (None): place holder
            target_3d (None): place holder
            weight (tensor or None): shape (n) or (n, keypoint_num*2)
        '''
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.reg_decoded_keypoints:
            pred = coder.decode(anchors, pred)
            weight = weight / self.image_resolution
        else:
            target_2d = coder.encode(anchors, target_2d)

        if self.loss_type == 'smooth_l1':
            loss = smooth_l1_loss(pred, target_2d, self.beta)
        elif self.loss_type == 'l1':
            loss = torch.abs(pred - target_2d)
        else:
            loss = torch.sqrt((pred - target_2d)**2)
        loss = self.loss_weight * weight_reduce_loss(
            loss, weight, reduction, avg_factor
        )
        return loss



@LOSSES.register_module()
class ObjectSpaceLoss(nn.Module):
    def __init__(self, 
                mesh_diameters, 
                loss_weight=1.0, 
                reduction='mean', 
                loss_type='l1',
                beta=None):
        super().__init__()
        self.mesh_diameters = nn.Parameter(torch.tensor(mesh_diameters).reshape(-1), requires_grad=False)
        self.beta = beta
        self.loss_weight = loss_weight
        self.reduction = reduction
        assert loss_type in ['smooth_l1', 'l1', 'l2']
        if loss_type == 'smooth_l1':
            assert isinstance(beta, float)
            self.beta = beta 
        self.loss_type = loss_type
    
    def object_space_loss(self, 
                        pred_2d, 
                        target_3d, 
                        inverse_internel_k, 
                        scaling_factor):
        '''
        Args:
            pred_2d (torch.Tensor): shape (N, 8, 2), xy format
            target_3d (torch.Tensor): shape (N, 8, 3), xyz format, 3d points in camera frame
            inverse_internel_k (torch.Tensor): shape (N, 3, 3)
            scaling_factor (torch.Tensor): shape (N), mesh diameteres used to scale the position
        return:
            loss (torch.Tensor): shape (N)
        '''
        num_preds, keypoints_num, _ = pred_2d.shape
        # shape (N, 8, 3, 3)
        inverse_internel_k = inverse_internel_k[:, None].expand(num_preds, keypoints_num, 3, 3)
        # shape (N, 8, 3, 1)
        preds_ext = torch.cat([pred_2d, pred_2d.new_ones(num_preds, keypoints_num, 1)], dim=-1)[..., None]

        # shape (N, 8, 3, 1)
        ray_direction = torch.matmul(inverse_internel_k, preds_ext)
        # shape (N, 8, 3, 3)
        projection_metric = torch.matmul(ray_direction, ray_direction.transpose(2, 3)) / \
                                    torch.matmul(ray_direction.transpose(2, 3), ray_direction)
        
        # shape (N*8, 3, 1)
        projected_3d_points = torch.matmul(projection_metric, target_3d[..., None])
        projected_3d_points = projected_3d_points.view(num_preds, keypoints_num, 3)

        # normalize by mesh diameter
        normalized_projected_3d_points = projected_3d_points / scaling_factor[..., None, None]
        normalized_target_3d = target_3d / scaling_factor[..., None, None]

        if self.loss_type == 'smooth_l1':
            loss = smooth_l1_loss(normalized_projected_3d_points, normalized_target_3d, beta=self.beta)
        elif self.loss_type == 'l1':
            loss = torch.abs(normalized_projected_3d_points - normalized_target_3d)
        elif self.loss_type == 'l2':
            loss = torch.sqrt((normalized_projected_3d_points - normalized_target_3d)**2)
        else:
            raise RuntimeError
        return loss

    def forward(self,
                pred,
                target_3d,
                label,
                internel_k,
                coder,
                anchors,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        '''
        Args:
            pred (tensor): shape (n, keypoint_num, 2), 2d prediction keypoints
            targets_3d (tensor): shape (n, keypoint_num, 3), 3d ground truch keypoints in camera frame
            internel_k (tensor): shape (n, 3, 3), 
            label (tensor): shape (n), assigned label
            target_2d (None): place holder
            weight (tensor, optioncal): shape (n) or (n, keypoint_num*2)
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        '''
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        # decode the predictions
        pred = coder.decode(anchors, pred)
        preds_num, keypoints_num = pred.size(0), pred.size(1)
        target_3d = target_3d.view(preds_num, keypoints_num, 3)
        mesh_diameters = self.mesh_diameters[label]
        inverse_internel_k = torch.linalg.inv(internel_k)
        loss = self.object_space_loss(
            pred,
            target_3d,
            inverse_internel_k=inverse_internel_k,
            scaling_factor=mesh_diameters, 
        )
        loss = self.loss_weight * weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss