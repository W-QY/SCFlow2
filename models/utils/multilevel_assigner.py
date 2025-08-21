import torch
from models.utils.utils import random_sample

INF = 1000000
class MultiLevelAssigner:
    def __init__(self,
                num_pos,
                pos_lambda,
                anchor_sizes,
                with_replacement=False,
                suppress_zero_level=False) -> None:
        self.num_pos = num_pos
        self.pos_lambda = pos_lambda
        self.anchor_sizes = torch.tensor(anchor_sizes).view(-1)
        self.suppress_zero_level = suppress_zero_level
        self.with_replacement = with_replacement
    
    def assign(self, points, gt_bboxes, gt_masks):
        '''
        args:
            points (list[tensor]): points across difference levels
            gt_bboxes (tenosr): shape (N, 4)
            gt_masks (tensor): shape (N, H, W)
        return:
            assigned_gt_inds (tensor): shape (n, N), n is for anchor points num
                0 means unassigned and -1 means ignore.
        '''

        assert len(points) == len(self.anchor_sizes)

        num_points_per_lvl = [point.size(0) for point in points]
        concat_points = torch.cat(points, dim=0)

        num_points = concat_points.size(0)
        num_gts = gt_bboxes.size(0)
        if num_gts == 0:
            return gt_bboxes.new_zeros((num_points, ), dtype=torch.int64)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        max_length, _ = torch.max(gt_bboxes[:, 2:] - gt_bboxes[:, :2], dim=1)

        if self.anchor_sizes.device != areas.device:
            self.anchor_sizes = self.anchor_sizes.to(areas.device)

        # compute sample num for each object on each level
        delta_k = torch.log2(max_length[..., None] / self.anchor_sizes[None])
        sample_ratio = torch.exp(-self.pos_lambda * torch.pow(delta_k, 2))
        sample_ratio = sample_ratio / sample_ratio.sum(dim=-1, keepdim=True)
        sample_num = (self.num_pos * sample_ratio + 0.5).int()
        if self.num_pos == -1:
            sample_num[...] = -1

        # shape (n, N); n for anchor points num
        areas = areas[None].repeat(num_points, 1)
        xs, ys = concat_points[:, 0].to(torch.int64), concat_points[:, 1].to(torch.int64)
        points_label = gt_masks[:, ys, xs]
        points_label = points_label.transpose(0, 1)

        inside_gt_mask = points_label == True
        areas[inside_gt_mask == 0] = INF
        
        min_area, min_area_inds = areas.min(dim=1)
        pos_flag = min_area < INF
        pos_inds = torch.nonzero(pos_flag).squeeze(-1)
        # 1-base, -1 means negative, 0 means ignore
        assigned_gt_inds = torch.zeros_like(min_area_inds) - 1
        assigned_gt_inds[pos_inds] = min_area_inds[pos_inds] + 1
        points_weights = torch.ones_like(min_area_inds, dtype=torch.float32)

        lvl_begin = 0
        total_ignore_inds = []
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            pos_inds_per_lvl = torch.nonzero(pos_flag[lvl_begin:lvl_end]).squeeze(-1)
            pos_assigned_gt_inds_per_lvl = min_area_inds[lvl_begin:lvl_end][pos_inds_per_lvl]
            sample_num_per_lvl = sample_num[:, lvl_idx]
            ignore_inds_list = []
            for i in range(num_gts):
                sample_num_per_lvl_per_obj = sample_num_per_lvl[i]
                if sample_num_per_lvl_per_obj == -1:
                    # activate all candidates, no ignored candidates
                    continue
                pos_inds_per_lvl_per_obj = pos_inds_per_lvl[torch.nonzero(pos_assigned_gt_inds_per_lvl == i).squeeze(-1)] + lvl_begin
                pos_num_per_lvl_per_obj = pos_inds_per_lvl_per_obj.size(0)
                if sample_num_per_lvl_per_obj == 0 and self.suppress_zero_level:
                    # negative sample
                    assigned_gt_inds[pos_inds_per_lvl_per_obj] = -1
                    continue
                if not self.with_replacement:
                    ignore_inds_list.append(random_sample(pos_inds_per_lvl_per_obj, pos_num_per_lvl_per_obj - sample_num_per_lvl_per_obj))
                else:
                    if pos_num_per_lvl_per_obj < sample_num_per_lvl_per_obj:
                        # all positive candidates are selected as positive samples, no ignored positive candidates
                        repeat_sample_time = sample_num_per_lvl_per_obj // pos_num_per_lvl_per_obj
                        points_weights[pos_inds_per_lvl_per_obj] = repeat_sample_time
                        again_sampled_inds = random_sample(pos_inds_per_lvl_per_obj, sample_num_per_lvl_per_obj%pos_num_per_lvl_per_obj)
                        points_weights[again_sampled_inds] += 1
                    else:
                        ignore_inds_list.append(random_sample(pos_inds_per_lvl_per_obj, pos_num_per_lvl_per_obj - sample_num_per_lvl_per_obj))

            total_ignore_inds.extend(ignore_inds_list)
            lvl_begin = lvl_end
        
        if len(total_ignore_inds) != 0:
            total_ignore_inds = torch.cat(total_ignore_inds)
            assigned_gt_inds[total_ignore_inds] = 0
            points_weights[total_ignore_inds] = 0

        return assigned_gt_inds, points_weights