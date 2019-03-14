from __future__ import division
import concurrent.futures
import h5py
import numpy as np
import utils
from pathlib import Path
import OpenEXR
import Imath
import glob

import os
import sys
import math
import numpy.random as random
import argparse

parser = argparse.ArgumentParser(description='dataset-processor')
parser.add_argument('--input', default='', type=str, metavar='PATH',
                    help='path to input folder')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to the output folder')


def convertSRGBToRGB(img_str, size):
    img = np.fromstring(img_str, dtype=np.float32)
    img = np.where(img <= 0.0031308,
                   (img * 12.92) * 255.0,
                   (1.055 * (img ** (1.0 / 2.4)) - 0.055) * 255.0)
    img.shape = (size[1], size[0])

    return img.astype(np.uint8)


def loadImageWithOpenEXR(filepath):
    image_exr = OpenEXR.InputFile(filepath)
    dw = image_exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    precision = Imath.PixelType(Imath.PixelType.FLOAT)
    Z = image_exr.channel('Z', precision)
    image_depth = np.fromstring(Z, dtype=np.float32)
    image_depth[image_depth > 99999999] = 0  # conversion: invalid depth in the exr is inf and on ros depth image is 0
    image_depth.shape = (size[1], size[0])

    r = convertSRGBToRGB(image_exr.channel('R', precision), size)
    g = convertSRGBToRGB(image_exr.channel('G', precision), size)
    b = convertSRGBToRGB(image_exr.channel('B', precision), size)
    rgb = np.stack([r, g, b])

    return rgb, image_depth


def saveH5(filepath, rgb, depth):
    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset("rgb_image_data", data=rgb, compression=4, chunks=(1, 60, 84),dtype='uint8')
        hf.create_dataset('dense_image_data', data=np.expand_dims(depth, axis=0), compression=4, chunks=(1, 60, 84),
                             dtype='float32')


def convertExr(input_file, output_file):
    bgr, depth = loadImageWithOpenEXR(input_file)
    saveH5(output_file, bgr, depth)


def retrive_sorted(folder):
    file_list = sorted([h5file for h5file in os.listdir(folder) if (h5file).endswith(".exr")])
    file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return file_list


if __name__ == "__main__":
    # setup the argument list
    parser = argparse.ArgumentParser(description='create a report about the quality of the training data')

    parser.add_argument('--input_folder', nargs='?', help='folder of folders with the .h5 files')
    parser.add_argument('--output_folder', nargs='?', help='output folder')

    # print help if no argument is specified
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    parsed = parser.parse_args()

    if parsed.input_folder is None:
        parser.print_help()
        sys.exit(0)

    if not os.path.exists(parsed.input_folder):
        print('The input_data_folder does not exist :' + parsed.input_folder)
        sys.exit(0)

    if not os.path.exists(parsed.output_folder):  # prevent a mess and to right on the input folder
        print('The output folder does not exist.')
        sys.exit(0)



    # subfolders = sorted([subfolder for subfolder in os.listdir(input_folder) if
    #                      os.path.isdir(os.path.join(input_folder, subfolder))])
    # print(subfolders)
    #
    #
    # for it_s, s in enumerate(subfolders):
    file_list = retrive_sorted(parsed.input_folder)

    for i, curr in enumerate(file_list):
        input_file = os.path.join(parsed.input_folder, curr)
        pre, ext = os.path.splitext(curr)
        output_file = os.path.join(parsed.output_folder, pre + '.h5')
        print('input file is:' + input_file)
        print('output file is:' + output_file + '\n')
        rgb, d = loadImageWithOpenEXR(input_file)
        #assert(rgb.shape == (3,480,752))
        saveH5(output_file,rgb,d)
