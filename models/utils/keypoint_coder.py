from typing import Optional, Sequence
import torch

def xy2offset(points, center, normalizer):
    '''
    Args:
        point (Tensor): shape (N, k, 2) xy format
        center (Tensor): shape (N, 2) xy format
        normalizer (Tensor): shape (N, 2), normalize the origin offset 
    '''
    assert isinstance(points, torch.Tensor)
    assert isinstance(center, torch.Tensor)
    assert isinstance(normalizer, torch.Tensor)
    assert center.size() ==  normalizer.size()
    _, k, _ = points.shape
    center = center[:, None].expand(1, k, 1)
    normalizer = normalizer[:, None].expand(1, k, 1)
    
    offset = (points - center) / normalizer
    return offset

def offset2xy(offset, center, normalizer, max_shape=None, clip_border=True):
    '''
    Args:
        offset (Tensor): shape (N, k, 2) xy format
        center (Tensor): shape (N, 2), xy format
        normalizer (Tensor): shape (B, N, 2) or (N, 2), normalize x and y respecitively
        max_shape (Tensor|list): the image shape (H, W)
    '''
    assert isinstance(normalizer, torch.Tensor)
    assert isinstance(center, torch.Tensor)
    assert isinstance(offset, torch.Tensor)
    assert center.size() == normalizer.size()

    _, k, _ = offset.shape 
    center = center[:, None].expand(-1, k, -1)
    normalizer = normalizer[:, None].expand(-1, k, -1)
    

    decoded_points = center + offset * normalizer
    if clip_border and max_shape is not None:
        min_xy = center.new_tensor(0)
        if not isinstance(max_shape, torch.Tensor):
            max_shape = center.new_tensor(max_shape)
        if max_shape.ndim == 2:
            # batch format
            assert offset.ndim == 3
            assert max_shape.size(0) == offset.size(0)
        max_shape = max_shape[..., :2].type_as(center)
        max_xy = torch.cat([max_shape]*k, dim=-1).flip(-1).unsqueeze(-2)
        decoded_points = torch.where(decoded_points < min_xy, min_xy, decoded_points)
        decoded_points = torch.where(decoded_points > max_xy, max_xy, decoded_points)
    return decoded_points
    


 

class TargetCoder():
    def __init__(self, normalizer=1.0, clip_border=False):
        self.normalizer = normalizer
        self.clip_border = clip_border
    

    def encode(self, bboxes:torch.Tensor, targets:torch.Tensor):
        '''
        Transform the ground truth keypoints into regrssion offsets according to the reference 'bbox', i.e. anchor
        Ground truth box can be also seen as two keypoints
        Args:
            bboxes (Tensor): Shape (N, 4)
            targets (Tensor): Shape (N, keypoint_num, 2) or (N, keypoint_num*2)
        '''
        assert bboxes.size(0) == targets.size(0)
        prior_centers = (bboxes[..., 0:2] + bboxes[..., 2:4])/2
        prior_normalizers = bboxes[..., 2:4] - bboxes[..., 0:2]
        if targets.ndim == 2:
            keypoint_num = targets.size(1) // 2
            targets = torch.stack([targets[..., :keypoint_num], targets[..., keypoint_num:]], dim=-1)
        
        _, keypoint_num, _ = targets.shape 
        center = prior_centers[:, None].expand(-1, keypoint_num, 2)
        normalizer = prior_normalizers[:, None].expand(-1, keypoint_num, 2)
        encoded_offset = (targets - center) / normalizer
        # encoded_offset = xy2offset(targets, prior_centers, prior_normalizers*self.normalizer)
        return encoded_offset
    
    def decode(self, bboxes:torch.Tensor, pred:torch.Tensor, max_shapes:Optional[Sequence[tuple]]=None):
        '''
        Transform the encoded offset(predicted) back to keypoints
        Args:
            bboxes (Tensor): Shape (N, 4).
            pred (Tensor): Shape (N, keypoint_num, 2) or (N, keypoint_num*2)
            max_shapes (Sequence[Sequence[int]] or Sequence[int] or torch.Tensor): 
                specifics the max_shape of decoded bbox or keypoints, (H, W) format
                if bboxes shape is (B, N, 4), then the max_shape should be Sequence[Sequence[int]]
        '''
        prior_centers = (bboxes[..., 0:2] + bboxes[..., 2:4])/2
        prior_normalizers = bboxes[..., 2:4] - bboxes[..., 0:2]
        if pred.ndim == 2:
            keypoint_num = pred.size(1) // 2
            pred_x = pred[..., :keypoint_num]
            pred_y = pred[..., keypoint_num:]
            pred = torch.stack([pred_x, pred_y], dim=-1)
        
        _, keypoint_num, _ = pred.shape 
        center = prior_centers[:, None].expand(-1, keypoint_num, -1)
        normalizer = prior_normalizers[:, None].expand(-1, keypoint_num, -1)
        decoded_points = center + pred * normalizer
        
        # if self.clip_border and max_shape is not None:
        #     min_xy = center.new_tensor(0)
        #     if not isinstance(max_shape, torch.Tensor):
        #         max_shape = center.new_tensor(max_shape)
        #     if max_shape.ndim == 2:
        #         # batch format
        #         assert offset.ndim == 3
        #         assert max_shape.size(0) == offset.size(0)
        #     max_shape = max_shape[..., :2].type_as(center)
        #     max_xy = torch.cat([max_shape]*k, dim=-1).flip(-1).unsqueeze(-2)
        #     decoded_points = torch.where(decoded_points < min_xy, min_xy, decoded_points)
        #     decoded_points = torch.where(decoded_points > max_xy, max_xy, decoded_points)
        return decoded_points