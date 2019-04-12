import os
import time
import csv
import numpy as np
import math
import torch
import sys
import torch.backends.cudnn as cudnn
from dataloaders.visim_dataloader import VISIMSeqDataset

def main():
    traindir = '/media/lucas/lucas-ds2-1tb/dataset_big_v8'
    valdir = '/media/lucas/lucas-ds2-1tb/dataset_big_v8'
    train_dataset = VISIMSeqDataset(traindir, type='train')
    val_dataset = VISIMSeqDataset(valdir, type='val')

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=True,
        num_workers=7, pin_memory=True, sampler=None,
        worker_init_fn=lambda work_id: np.random.seed(work_id))

    for result_input_tensor,result_target_tensor,result_scales,result_transforms in train_loader:
        print(len(result_transforms))

        #assert(result_input_tensor)



if __name__ == '__main__':
    main()