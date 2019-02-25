import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()


    def forward(self, pred, target_depth):
        #target_depth = target[:,0:1,:,:]
        assert pred.dim() == target_depth.dim(), "inconsistent dimensions"
        valid_mask = ((target_depth>0).detach())

        num_valids = valid_mask.sum()
        if num_valids < 10:
            return None

        diff = target_depth - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        num_valids = valid_mask.sum()
        if num_valids < 10:
            return None

        diff = target - pred
        # print('std - min:{} | max:{} | mean:{} '.format(diff.min(),diff.max(),diff.mean()))
        # print('mask size:{} '.format(valid_mask.sum()))
        diff = diff[valid_mask]
        # print('mask - min:{} | max:{} | mean:{} '.format(diff.min(), diff.max(), diff.mean()))
        # diff = diff ** 2
        # print('pow2 - min:{} | max:{} | mean:{} '.format(diff.min(), diff.max(), diff.mean()))

        self.loss = diff.abs().mean() # diff.mean() #
        #print(self.loss)
        return self.loss