import torch
import math
import numpy as np

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.loss0, self.loss1,self.loss2 = 0, 0,0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.loss0, self.loss1, self.loss2 = np.inf, np.inf, np.inf

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time,loss0,loss1,loss2):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time
        self.loss0, self.loss1, self.loss2 = loss0,loss1,loss2

    def evaluate(self, output, target):
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

class ConfidencePixelwiseAverageMeter(object):
    def __init__(self,num_bins=1000):
        self.num_bins = num_bins
        self.reset()

    def reset(self):
        self.count = np.zeros([self.num_bins], np.uint64)
        self.absrel = np.zeros([self.num_bins], np.uint64)

    def hash_index(self,confidence): #confidence is a matrix between 0 and 1
        indexs = np.floor((confidence - 10e-15) * self.num_bins)
        return indexs.astype(int)

    def evaluate(self, depth, confidence, target):
        valid_mask = target > 0
        depth = depth[valid_mask]
        target = target[valid_mask]
        confidence = confidence[valid_mask]
        indexes = self.hash_index(confidence.cpu().numpy())

        abs_diff = (depth - target).abs()
        absrel = ((abs_diff / target)*1000).cpu().numpy()

        for img_index, conf_index in np.ndenumerate(indexes):
            self.count[conf_index] += 1
            self.absrel[conf_index] += absrel[img_index]

    def result(self):
        res = [None] * self.num_bins
        for pos, conf_index in np.ndenumerate(self.count):
            if self.count[pos] > 0:
                res[pos[0]] = self.absrel[pos]/ self.count[pos]
        return res





class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0
        self.sum_loss0, self.sum_loss1, self.sum_loss2 = 0,0,0

    def update(self, result, gpu_time, data_time,loss, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time
        self.sum_loss0 += n * loss[0]
        self.sum_loss1 += n * loss[1]
        self.sum_loss2 += n * loss[2]

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count, self.sum_loss0 / self.count, self.sum_loss1 / self.count, self.sum_loss2 / self.count)
        return avg