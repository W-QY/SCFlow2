from typing import Optional
import torch 



class Procrustes:
    @classmethod
    def residuals(self, X:torch.Tensor, Y:torch.Tensor, rot:torch.Tensor, trans:torch.Tensor):
        Y_est = torch.matmul(rot, X.t()).t() + trans[None]
        return torch.linalg.norm(Y_est - Y, dim=-1, ord=2)

    @classmethod
    def estimate(self, X:torch.Tensor, Y:torch.Tensor):
        X_c = torch.mean(X, dim=0)
        Y_c = torch.mean(Y, dim=0)

        X_s = X - X_c
        Y_s = Y - Y_c 

        c = torch.matmul(Y_s.t(), X_s)

        U, S, V = torch.linalg.svd(c)
        d = torch.linalg.det(U) * torch.linalg.det(V)
        
        if d < 0:
            S[-1] = -S[-1]
            U[:, -1] = -U[:, -1]
        
        rot = torch.matmul(U, V)
        trans = Y_c - torch.matmul(rot, X_c)
        return rot, trans

class RansacEstimator:
    def __init__(self, 
                min_samples:Optional[int]=None,
                residual_threshold:Optional[float]=None,
                max_trials:Optional[int]=200) -> None:
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trails = max_trials
    
    def fit(self, X:torch.Tensor, Y:torch.Tensor):
        best_inliers, best_num_inliers = None, 0
        num_data, num_feats = X.shape
        device = X.device
        for _ in range(self.max_trails):
            # randomly select subset
            rand_subset_idxs = torch.randperm(num_data, device=device)[:self.min_samples]
            rand_subset_X, rand_subset_Y = X[rand_subset_idxs], Y[rand_subset_idxs]
            # estimate with model
            rot, trans = Procrustes.estimate(rand_subset_X, rand_subset_Y)
            # compute residuals
            residuals = Procrustes.residuals(X, Y, rot, trans)
            # judge inliers and outliers
            inliers = residuals <= self.residual_threshold
            num_inliers = torch.sum(inliers)
            # decide if better
            if (best_num_inliers < num_inliers):
                best_num_inliers = num_inliers
                best_inliers = inliers
        
        rot, trans = Procrustes.estimate(X[best_inliers], Y[best_inliers])
        ret = {
            "best_params": [rot, trans],
            "best_inliers": best_inliers,
            "best_num_inliers": best_num_inliers,
            "retval":True,
        }
        return ret


class BatchProcustes:
    
    @classmethod
    def residuals(self, X:torch.Tensor, Y:torch.Tensor, rot:torch.Tensor, trans:torch.Tensor):
        '''
        args:
            X: shape (N, 3)
            Y: shape (N, 3)
            rot: shape (3, 3)
            trans: shape (3)
        '''
        Y_est = torch.matmul(rot, X.t()).transpose(1, 2) + trans[:, None]
        return torch.linalg.norm(Y_est - Y, dim=-1, ord=2)

    @classmethod
    def estimate(self, X:torch.Tensor, Y:torch.Tensor):
        '''
        X:shape (trials, min_samples, 3)
        Y:shape (trials, min_samples, 3)
        '''
        X_c = torch.mean(X, dim=1, keepdim=True)
        Y_c = torch.mean(Y, dim=1, keepdim=True)

        X_s, Y_s = X - X_c, Y - Y_c
        c = torch.matmul(Y_s.transpose(2, 1), X_s)

        U, S, V = torch.linalg.svd(c)
        d = (torch.linalg.det(U) * torch.linalg.det(V)) < 0
        S[d, -1] = -S[d, -1]
        U[d, :, -1] = -U[d, :, -1]
    
        rot = torch.matmul(U, V)
        trans = Y_c - torch.matmul(rot, X_c.transpose(2, 1)).transpose(2, 1)
        return rot, trans.squeeze(dim=1)



class BatchRansacEstimator:
    def __init__(self, 
                min_samples:Optional[int]=None,
                residual_threshold:Optional[float]=None,
                max_trials:Optional[int]=200,
                max_inliers:Optional[int]=5000) -> None:
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trails = max_trials
        self.max_inliers = max_inliers
    
    def fit(self, X:torch.Tensor, Y:torch.Tensor):
        num_data, num_feats = X.shape
        device = X.device
        if self.min_samples * self.max_trails <= num_data:
            rand_subset_idxs = torch.randperm(num_data, device=device)[:self.min_samples*self.max_trails]
        else:
            return {'retval':False}
        rand_subset_X = X[rand_subset_idxs].reshape(self.max_trails, self.min_samples, num_feats)
        rand_subset_Y = Y[rand_subset_idxs].reshape(self.max_trails, self.min_samples, num_feats)

        all_trials_rot, all_trials_trans = BatchProcustes.estimate(rand_subset_X, rand_subset_Y)

        residuals = BatchProcustes.residuals(X, Y, all_trials_rot, all_trials_trans)
        inliers = residuals < self.residual_threshold
        num_inliers = torch.sum(inliers, dim=-1)
        best_num_inliers, best_trial_index = torch.max(num_inliers, dim=0)
        best_inliers = inliers[best_trial_index]
        rot, trans = Procrustes.estimate(X[best_inliers], Y[best_inliers])
        return {
            "best_params": [rot, trans],
            "best_inliers": best_inliers,
            "best_num_inliers": best_num_inliers,
            "retval": True
        }