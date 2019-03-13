import os
import sys
from dataloaders.dataloader_ext import MyDataloaderExt
import concurrent.futures
import h5py
import numpy as np
import utils
import math
from pathlib import Path

import argparse
parser = argparse.ArgumentParser(description='dataset-processor')
parser.add_argument('--input', default='', type=str, metavar='PATH',
                    help='path to input folder')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to the output folder')

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")

    traindir = args.input
    valdir = args.input

    train_dataset = MyDataloaderExt(traindir, type='train')
    val_dataset = MyDataloaderExt(valdir, type='val')

    print("=> data loaders created.")
    return train_dataset, val_dataset

def process_sample(dataset,output,index):
    path, target = dataset.imgs[index]
    print('processing {} - {}/{}'.format(path,index+1,dataset.__len__()))

    with h5py.File(path, "r") as h5f:

        dense_data = h5f['dense_image_data']
        depth = np.array(dense_data[0:1, :, :])
        mask_array = depth > 10000  # in this software inf distance is zero.
        depth[mask_array] = 0

        new_dense_data = np.array(depth, copy=True)

        from pathlib import Path
        old_path = Path(path)
        new_folder = os.path.split(old_path.parent)[1]
        os.makedirs(os.path.join(output,new_folder),exist_ok=True)
        out_file = os.path.join(output,new_folder,old_path.name)

        with h5py.File(out_file, "w") as h5out:
            h5f.copy('landmark_2d_data',h5out)
            h5f.copy('gt_twc_data', h5out)
            h5f.copy('slam_twc_data', h5out)
            h5f.copy('timestamp', h5out)
            h5f.copy('rgb_image_data', h5out)
            h5out.create_dataset('dense_image_data', data=new_dense_data, compression=4, chunks=(1, 60, 84),
                                 dtype='float32')
            h5out.close()
        h5f.close()

        return out_file


def process_dataset(dataset,output_directory,parallel=True):
    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thread_res = {executor.submit(process_sample,dataset,output_directory,  n): n for n in range(dataset.__len__())}
            for res in concurrent.futures.as_completed(thread_res):
                id = thread_res[res]
                print('Done:%s' % id)
                # try:
                #     data = id
                # except Exception as exc:
                #     print('%r generated an exception: %s' % (id, exc))
                # else:
                #     print('%r Done:%s' % id)
    else:
        for index in range(dataset.__len__()):
            process_sample(dataset,output_directory,index)


def main():

    args = parser.parse_args()

    #print help if no argument is specified
    if len(sys.argv)<2:
        parser.print_help()
        sys.exit(0)


    output_directory = args.output
    training_loader, val_loader = create_data_loaders(args)

    os.makedirs(output_directory, exist_ok=True)
    if os.path.isdir(output_directory) : #and not os.path.exists(output_directory)
        process_dataset(training_loader,output_directory)
        process_dataset(val_loader,output_directory)
    else:
        print('error')


    return

if __name__ == '__main__':
    main()
