import os
import os.path
import numpy as np
import torch.utils.data as data
import torch
import h5py
import csv 
import dataloaders.transforms as transforms
import math
import argparse

IMG_EXTENSIONS = ['.h5',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_files_list(path):
    images = []    
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        folder = os.path.dirname(path)
        for row in reader:            
            path = os.path.join(folder, row[1])
            item = (path,row[0])
            images.append(item)
    return images

def rgb2grayscale(rgb):
    return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

class Modality():
    modality_names = ['rgb', 'grey', 'fd', 'kor', 'kgt', 'kw', 'kde', 'dor', 'dde', 'kvor', 'd2dwor', 'd3dwde','d3dwor','wkde','wdde']
    depth_channels_names = ['fd', 'kor', 'kde', 'kgt', 'dor', 'dde', 'kvor']
    weight_names = ['d2dwor', 'd3dwde','d3dwor', 'kw','wkde','wdde']
    #color_names = ['rgb', 'grey']

    def __init__(self, value):
        self.modalities = value.split('-')
        self.is_valid = self.validate()
        if not self.is_valid:
            self.modalities = []

        self.format = self.calc_format()


    def __contains__(self, key):
        return key in self.modalities

    # def num_channels(self):
    #     num = len(self.modalities) # remove groundtruth channel
    #     if 'rgb' in self.modalities:
    #         num = num+2
    #     return num

    def num_channels(self):
        num_channel = len(self.format)
        return num_channel

    def validate(self):
        for token in self.modalities:
            if not token in self.modality_names:
                print('token: "{}" is invalid'.format())
                return False
        return True

    def get_input_image_channel(self):
        if 'rgb' in self.modalities:
            return 3,'rgb'

        if 'grey' in self.modalities:
            return 1,'grey'

        return 0, ''

    def get_input_depth_channel(self):
        for token in self.modalities:
            if token in self.depth_channels_names:
                return 1,token
                break
        return 0, ''

    def get_input_weight_channel(self):
        for token in self.modalities:
            if token in self.weight_names:
                return 1,token
                break
        return 0,''


    def calc_format(self):
        format_out =''
        if 'rgb' in self.modalities:
            format_out = format_out + 'rgb'

        if 'grey' in self.modalities:
            format_out = format_out + 'g'

        for token in self.modalities:
            if  token in self.depth_channels_names:
                format_out = format_out + 'd'
                break

        for token in self.modalities:
            if  token in self.weight_names:
                format_out = format_out + 'w'
                break

        return format_out

    @staticmethod
    def validate_static(value):
        modals = value.split('-')
        for token in modals:
            if not token in Modality.modality_names:
                raise argparse.ArgumentTypeError("%s is an invalid channel" % token)
                return False
        return True

class MyDataloaderExt(data.Dataset):

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        
        imgs = []
        if type == 'train':
            self.transform = self.train_transform
            imgs = load_files_list(os.path.join(root,'train.txt'))
        elif type == 'val':
            self.transform = self.val_transform
            imgs = load_files_list(os.path.join(root,'val.txt'))
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))

        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
#        self.classes = classes
#        self.class_to_idx = class_to_idx

        self.sparsifier = sparsifier
        self.modality = Modality(modality)



    def train_transform(self, channels):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, channels):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, targe_depth):
        if self.sparsifier is None:
            raise (RuntimeError("please select a sparsifier "))
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, targe_depth)
            sparse_depth = np.zeros(targe_depth.shape)
            sparse_depth[mask_keep] = targe_depth[mask_keep]
            return sparse_depth


    # gt_depth - gt depth
    # rgb -color channel
    # grey - disable
    # fd - fake slam uing the given sparsifier from gt depth
    # kor - slam keypoint + slam depth
    # kde - slam keypoint + mesh-based denoise depth
    # kgt - slam keypoint + gt depth
    # kw - sparse confidence measurements
    # dor - mesh+interpolation of slam points
    # dde - mesh+interpolation of denoised slam points
    # kvor - slam keypoint  expanded to voronoi diagram cell around (dense)

    # d2dwor - 2d image distance transformantion using slam keypoints as seeds
    # d3dwde - 3d euclidian distance to closest the denoised slam keypoint
    # d3dwor - 3d euclidian distance to closest the slam keypoint

    def h5_loader_general(self,index,type):
        result = dict()
        path, target = self.imgs[index]
        h5f = h5py.File(path, "r")

        #target depth
        dense_data = h5f['dense_image_data']
        depth = np.array(dense_data[0, :, :])
        mask_array = depth > 10000 # in this software inf distance is zero.
        depth[mask_array] = 0
        result['gt_depth'] = depth

        # color data

        rgb = np.array(h5f['rgb_image_data'])
        rgb = np.transpose(rgb, (1, 2, 0))

        if 'rgb' in type:
            result['rgb'] = rgb

        if 'grey' in type:
            result['grey'] = rgb2grayscale(rgb)

        #fake sparse data using the spasificator and ground-truth depth
        if 'fd' in type:
            result['fd'] = self.create_sparse_depth(rgb, depth)

        #using real keypoints from slam
        data_2d = np.array(h5f['landmark_2d_data'])

        if 'kor' in type:
            kor_input = np.zeros_like(depth)
            for row in data_2d:
                xp = int(math.floor(row[1]))
                yp = int(math.floor(row[0]))
                if (row[2] > 0):
                    kor_input[xp, yp] = row[2]
            result['kor'] = kor_input

        if 'kgt' in type:
            kgt_input = np.zeros_like(depth)
            for row in data_2d:
                xp = int(math.floor(row[1]))
                yp = int(math.floor(row[0]))
                if (depth[xp, yp] > 0):
                    kgt_input[xp, yp] = depth[xp, yp]
            result['kgt'] = kgt_input

        if 'kde' in type:
            kde_input = np.zeros_like(depth)
            for row in data_2d:
                xp = int(math.floor(row[1]))
                yp = int(math.floor(row[0]))
                if (row[3] > 0):
                    kde_input[xp, yp] = row[3]
            result['kde'] = kde_input

        if 'wkde' in type:
            kde_input = np.zeros_like(depth)
            for row in data_2d:
                xp = int(math.floor(row[1]))
                yp = int(math.floor(row[0]))
                if (row[3] > 0):
                    kde_input[xp, yp] = row[3]
            result['wkde'] = kde_input


        if 'kw' in type:
            kw_input = np.zeros_like(depth)
            for row in data_2d:
                xp = int(math.floor(row[1]))
                yp = int(math.floor(row[0]))
                if (row[4] > 0):
                    kw_input[xp, yp] = row[4]
            result['kw'] = kw_input

        if 'kvor' in type:
            raise (RuntimeError("transform not defined"))
        if 'd2dwor' in type:
            raise (RuntimeError("transform not defined"))

        if 'dor' in type or 'dde' in type or 'd3dwor' in type or 'd3dwde' in type or 'wdde' in type:
            dense_data = h5f['dense_image_data']

            if 'dor' in type:
                result['dor'] = np.array(dense_data[1, :, :])

            if 'dde' in type:
                result['dde'] = np.array(dense_data[4, :, :])

            if 'd3dwor' in type:
                result['d3dwor'] = np.array(dense_data[3, :, :])

            if 'd3dwde' in type:
                result['d3dwde'] = np.array(dense_data[6, :, :])

            if 'wdde' in type:
                result['wdde'] = np.array(dense_data[4, :, :])


        return result

    def to_tensor(self, img):

        if not isinstance(img, np.ndarray):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        # handle numpy array
        if img.ndim == 3 or img.ndim == 2:
            img = torch.from_numpy(img.copy())
        else:
            raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

        return img.float()

    def append_tensor3d(self,input_np,value):
        if not isinstance(input_np, np.ndarray):  # first element
            if value.ndim == 2:
                input_np = np.expand_dims(value, axis=0)
            elif value.ndim == 3:
                input_np = value
        else:  # 2nd ,3rd ...
            if value.ndim == 2:
                input_np = np.append(input_np, np.expand_dims(value, axis=0), axis=0)
            elif value.ndim == 3:
                input_np = np.append(input_np, value, axis=0)
            else:
                raise RuntimeError('value should be ndarray with 2 or 3 dimensions. Got {}'.format(value.ndim))
        return input_np

    def __getitem__(self, index):

        input_np = None

        channels_np = self.h5_loader_general(index, self.modality)

        if self.transform is not None:
            channels_transformed_np = self.transform(channels_np)
        else:
            raise (RuntimeError("transform not defined"))

        for key, value in channels_transformed_np.items():
            if key == 'gt_depth':
                continue

        num_image_channel,image_channel = self.modality.get_input_image_channel()
        if num_image_channel > 0:
            input_np = self.append_tensor3d(input_np,channels_transformed_np[image_channel])

        num_depth_channel, depth_channel = self.modality.get_input_depth_channel()
        if num_depth_channel > 0:
            input_np = self.append_tensor3d(input_np, channels_transformed_np[depth_channel])

        num_weight_channel, weight_channel = self.modality.get_input_weight_channel()
        if num_weight_channel > 0:
            input_np = self.append_tensor3d(input_np, channels_transformed_np[weight_channel])

        input_tensor = self.to_tensor(input_np)
        target_depth_tensor = self.to_tensor(channels_transformed_np['gt_depth']).unsqueeze(0)


        #target_depth_tensor = depth_tensor.unsqueeze(0)
        #        if input_tensor.dim() == 2: #force to have a third dimension on the single channel input
#            input_tensor = input_tensor.unsqueeze(0)
#        if input_tensor.dim() < 2:
#            raise (RuntimeError("transform not defined"))
 #       depth_tensor = totensor(depth_np)
#        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, target_depth_tensor

    def __len__(self):
        return len(self.imgs)
