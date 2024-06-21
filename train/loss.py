import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
from options import Options
from pythonUtils import *

sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

class ScoreLoss(nn.Module):
    def __init__(self, opt: Options, level: int):
        super(ScoreLoss, self).__init__()
        self.opt = opt
        self.n_classes = opt.classes
        self.scoreLevel = level
        self.alpha = 10
        self.k = 0

        scores = np.arange(self.scoreLevel).reshape(-1, 1)
        self.scores = torch.tensor(np.tile(scores, (1, self.n_classes)), dtype=opt.dtype, device=opt.device).unsqueeze(0)
        
        self.gt_true = self._norm_func(np.linspace(0, 1, self.scoreLevel), 0.8, 0.2)
        self.gt_true = (self.gt_true / np.sum(self.gt_true)).reshape(1, -1, 1)
        
        self.gt_false_mean = (0, 0.5)
        self.gt_false_std = (0.6, 1)
        
    def _norm_func(self, x, u, sig):
        return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)

    def forward(self, logits, targets: torch.Tensor):
        batch_size = targets.size(0)
        gt = np.zeros((batch_size, self.scoreLevel, self.n_classes))

        for j in range(batch_size):
            for i in range(self.n_classes):
                if targets[j].item() == i:
                    gt[j, :, i] = self.gt_true[:, 0]
                else:
                    gt_false = self._norm_func(
                        np.linspace(0, 1, self.scoreLevel),
                        random.uniform(*self.gt_false_mean),
                        random.uniform(*self.gt_false_std)
                    )
                    gt[j, :, i] = (gt_false / np.sum(gt_false)).reshape(-1)
        
        gt = torch.tensor(gt, dtype=self.opt.dtype, device=self.opt.device)
        logits = logits.to(dtype=self.opt.dtype)
        
        pred_scores = torch.mul(self.scores, logits)
        gt_scores = torch.mul(self.scores, gt)
        
        score_dist = torch.sqrt(torch.sum(torch.square(pred_scores - gt_scores)))
        mean_pred_score = torch.mean(pred_scores)
        loss = score_dist + self.alpha * self.k * (1 / mean_pred_score + mean_pred_score)
        
        return loss, score_dist, mean_pred_score

    def print_info(self, file=None):
        print_Extension('\nLoss Function Information: ScoreLoss', _file=file)
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            key_str = f"{str(key).capitalize()}:".center(30, " ")
            value_str = str(value) if not isinstance(value, (torch.Tensor, np.ndarray)) else f'\n{value}'
            print_Extension(key_str + value_str, _file=file)

if __name__ == '__main__':
    import train_config as config
    opt = Options(config)
    opt.classes = 10
    loss_fn = ScoreLoss(opt, 10)
    loss_fn.print_info()
