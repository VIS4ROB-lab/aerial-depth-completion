import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torchsummary import summary


from models import ResNet
from model_ext import DepthCompletionNet
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils
from dataloaders.dataloader_ext import Modality
from dataloaders.visim_dataloader import VISIMDataset

import argparse
parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('--data-path', default='data', type=str, metavar='PATH',
                    help='path to data folder')
args = parser.parse_args()
args.depth_divider =1.0


g_modality = Modality(args.modality)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")

    traindir = args.data_path
    valdir = args.data_path


    train_dataset = VISIMDataset(traindir, type='train',
            modality=args.modality,depth_divider=args.depth_divider)
    val_dataset = VISIMDataset(valdir, type='val',
        modality=args.modality,depth_divider=args.depth_divider)
    print("=> data loaders created.")
    return train_dataset, val_dataset

def main():
    global args, best_result, output_directory, train_csv, test_csv

    output_directory = os.path.dirname(args.evaluate)
    training_loader, val_loader = create_data_loaders(args)

    dataset_stats(training_loader)
    dataset_stats(val_loader)
    return





def dataset_stats(dataset_loader, write_to_file=True):
    average_meter = AverageMeter()
    all_modalities = Modality('kor--kde--dor--dde')
    for i in range(dataset_loader.__len__()):
        images = dataset_loader.h5_loader_general(i,all_modalities)

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data*args.depth_divider, target.data*args.depth_divider)
        average_meter.update(result, 0, 0, input.size(0))

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()
