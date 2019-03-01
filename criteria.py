import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) #scale back to real values
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


# get_gradient = Sobel()
#
#
# def calc_torch_normalmap(depth_np):
#     depth_tensor = torch.from_numpy(depth_np.copy()).unsqueeze(0).unsqueeze(0).float()
#     print(depth_tensor.shape)
#
#     depth_grad = get_gradient(depth_tensor)
#
#     depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth_tensor)
#     depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth_tensor)
#
#     ones = torch.ones(depth_tensor.size(0), 1, depth_tensor.size(2), depth_tensor.size(3)).float()
#     ones = torch.autograd.Variable(ones)
#
#     depth_normal = torch.cat((-depth_grad_dx/8, -depth_grad_dy/8, ones), 1)
#
#     depth_normal = (F.normalize(depth_normal, p=2, dim=1) + 1)/2
#
#     print(depth_normal.shape)
#
#     return depth_normal
#
# class GradLoss(nn.Module):
#     def __init__(self):
#         super(GradLoss, self).__init__()
#
#     # L1 norm
#     def forward(self, grad_fake, grad_real):
#         return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))
#
#
# class NormalLoss(nn.Module):
#     def __init__(self):
#         super(NormalLoss, self).__init__()
#
#     def forward(self, grad_fake, grad_real):
#         prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
#         fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
#         real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))
#
#         return 1 - torch.mean(prod / (fake_norm * real_norm))

class MaskedL2GradNormalLoss(nn.Module):

    def __init__(self):
        super(MaskedL2GradNormalLoss, self).__init__()
        self.get_gradient = Sobel()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        for param in self.parameters():
            param.requires_grad = False

    def get_extra_visualization(self):
        return self.output_normal,self.depth_normal

    def forward(self, pred, target_depth,epoch=None):

        assert pred.dim() == target_depth.dim(), "inconsistent dimensions"
        valid_mask = ((target_depth > 0).detach())

        num_valids = valid_mask.sum()
        if num_valids < 10:
            return None


        diff = target_depth - pred
        diff = diff[valid_mask]

        depth_grad = self.get_gradient(target_depth)
        output_grad = self.get_gradient(pred)

        loss_depth = torch.log(torch.abs(diff) + 0.5).mean()

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(target_depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(target_depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(target_depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(target_depth)

        ones = torch.ones(target_depth.size(0), 1, target_depth.size(2), target_depth.size(3)).cuda().float()

        self.depth_normal = torch.cat((-depth_grad_dx/8, -depth_grad_dy/8, ones), 1)
        self.output_normal = torch.cat((-output_grad_dx/8, -output_grad_dy/8, ones), 1)



        #self.depth_normal = F.normalize(self.depth_normal, p=2, dim=1)
        #self.output_normal = F.normalize(self.output_normal, p=2, dim=1)

        loss_dx = torch.log(torch.abs((output_grad_dx - depth_grad_dx)[valid_mask]) + 0.5).mean()
        loss_dy = torch.log(torch.abs((output_grad_dy - depth_grad_dy)[valid_mask]) + 0.5).mean()
        loss_normal = 10* torch.abs(1 - self.cos(self.output_normal, self.depth_normal)[valid_mask[:,0,:,:]]).mean()

        self.loss = [loss_depth.cpu().detach().numpy() , loss_normal.cpu().detach().numpy() , (loss_dx + loss_dy).cpu().detach().numpy()]

        final_loss = loss_depth + loss_normal

        return final_loss





class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()


    def get_extra_visualization(self):
        return None,None

    def forward(self, pred, target_depth,epoch=None):
        #target_depth = target[:,0:1,:,:]
        assert pred.dim() == target_depth.dim(), "inconsistent dimensions"
        valid_mask = ((target_depth>0).detach())

        num_valids = valid_mask.sum()
        if num_valids < 10:
            return None

        diff = target_depth - pred
        diff = diff[valid_mask]
        final_loss = (diff ** 2).mean()

        self.loss = [final_loss.cpu().detach().numpy(),0,0]

        return final_loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def get_extra_visualization(self):
        return None,None

    def forward(self, pred, target,epoch=None):
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

        final_loss = diff.abs().mean()
        self.loss = [final_loss.cpu().detach().numpy(),0,0] # diff.mean() #
        #print(self.loss)
        return final_loss