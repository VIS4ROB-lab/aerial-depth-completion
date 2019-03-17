import os
import os.path
import torch.utils.data as data
import torch
import h5py
import csv 
import dataloaders.transforms as transforms
import math
import argparse
from scipy import ndimage
import numpy as np
#import cv2 leak memory. try to avoid


epsilon= np.finfo(float).eps

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
    return rgb[0,:,:] * 0.2989 + rgb[1,:,:] * 0.587 + rgb[2,:,:] * 0.114

class Modality():
    #modality_names = ['rgb', 'grey', 'fd', 'kor', 'kgt', 'kw', 'kde', 'dor', 'dde', 'dvor', 'd2dwor', 'd3dwde','d3dwor','wkde','wdde']

    depth_channels_names = ['fd', 'kor', 'kde', 'kgt', 'dor', 'dde', 'dvor', 'dvgt', 'dvde','dore','ddee']
    metric_weight_names = ['d3dwde','d3dwor','wkde','wdde']
    image_size_weight_names = ['d2dwde','d2dwgt','d2dwor']
    weight_names = image_size_weight_names + ['kw'] + metric_weight_names
    need_divider = depth_channels_names + ['gt_depth'] + metric_weight_names
    color_names = ['rgb', 'grey']
    modality_names = color_names+depth_channels_names+weight_names

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

    def calc_from_sparse_input(self,in_sparse_map,voronoi=True,edt=True):

        res_voronoi = None
        res_edt = None

        if voronoi or edt: 
            mask = (in_sparse_map < epsilon)
            edt_result = ndimage.distance_transform_edt(mask, return_indices=voronoi)
            res_edt = np.sqrt(edt_result[0])

            if voronoi:
                res_voronoi = np.zeros_like(in_sparse_map)
                it = np.nditer(res_voronoi, flags=['multi_index'], op_flags=['writeonly'])

                with it:
                    while not it.finished:
                        xp = edt_result[1][0, it.multi_index[0], it.multi_index[1]]
                        yp = edt_result[1][1, it.multi_index[0], it.multi_index[1]]

                        it[0] = in_sparse_map[xp, yp]
                        it.iternext()

        return res_voronoi,res_edt

 #   def h5_preprocess(self,index):


    def h5_loader_general(self,index,type):
        result = dict()
        path, target = self.imgs[index]
        h5f = h5py.File(path, "r")

        #target depth
        if 'dense_image_data' in h5f:
            dense_data = h5f['dense_image_data']
            depth = np.array(dense_data[0, :, :])
            mask_array = depth > 10000 # in this software inf distance is zero.
            depth[mask_array] = 0
            result['gt_depth'] = depth
            if 'normal_data' in h5f:
                normal_rescaled = ((np.array(h5f['normal_data'],dtype='float32')/127.5) - 1.0)
                result['normal_x'] = normal_rescaled[0,:,:]
                result['normal_y'] = normal_rescaled[1, :, :]
                result['normal_z'] = normal_rescaled[2, :, :]
        elif 'depth' in h5f:
            result['gt_depth'] = depth = np.array(h5f['depth'])


        # color data
        if 'rgb_image_data' in h5f:
            rgb = np.array(h5f['rgb_image_data'])
        elif 'rgb' in h5f:
            rgb = np.array(h5f['rgb'])
        else:
            rgb = None


        if 'grey' in type:
            grey_img = rgb2grayscale(rgb)
            result['grey'] = grey_img

        rgb = np.transpose(rgb, (1, 2, 0))

        if 'rgb' in type:
            result['rgb'] = rgb

        #fake sparse data using the spasificator and ground-truth depth
        if 'fd' in type:
            result['fd'] = self.create_sparse_depth(rgb, depth)

        #using real keypoints from slam
        if 'landmark_2d_data' in h5f:
            data_2d = np.array(h5f['landmark_2d_data'])
        else:
            data_2d = None



        if 'kor' in type:
            kor_input = np.zeros_like(depth)
            for row in data_2d:
                xp = int(math.floor(row[1]))
                yp = int(math.floor(row[0]))
                if (row[2] > 0):
                    kor_input[xp, yp] = row[2]

            #res_voronoi,res_edt = self.calc_from_sparse_input(kor_input,'dvor' in type,'d2dwor' in type)

            if 'kor' in type:
                result['kor'] = kor_input
            # if 'dvor' in type:
            #     result['dvor'] = res_voronoi
            # if 'd2dwor' in type:
            #     result['d2dwor'] = res_edt


        if 'kgt' in type or 'dvgt' in type or 'd2dwgt' in type:
            kgt_input = np.zeros_like(depth)
            for row in data_2d:
                xp = int(math.floor(row[1]))
                yp = int(math.floor(row[0]))
                if (depth[xp, yp] > 0):
                    kgt_input[xp, yp] = depth[xp, yp]
            res_voronoi, res_edt = self.calc_from_sparse_input(kgt_input, 'dvgt' in type,'d2dwgt' in type)

            if 'kgt' in type:
                result['kgt'] = kgt_input
            if 'dvgt' in type:
                result['dvgt'] = res_voronoi
            if 'd2dwgt' in type:
                result['d2dwgt'] = res_edt

        if 'kde' in type or 'dvde' in type or 'd2dwde' in type:
            kde_input = np.zeros_like(depth)
            for row in data_2d:
                xp = int(math.floor(row[1]))
                yp = int(math.floor(row[0]))
                if (row[3] > 0):
                    kde_input[xp, yp] = row[3]

            res_voronoi, res_edt = self.calc_from_sparse_input(kde_input, 'dvde' in type,'d2dwde' in type)

            if 'kde' in type:
                result['kde'] = kde_input
            if 'dvde' in type:
                result['dvde'] = res_voronoi
            if 'd2dwde' in type:
                result['d2dwde'] = res_edt

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


        if 'dor' in type:
            result['dor'] = np.array(dense_data[1, :, :])

        if 'dore' in type:
            result['dore'] = np.array(dense_data[1, :, :])
            dore_mask = result['dore'] < epsilon
            result['dore'][dore_mask] = np.array(dense_data[2, :, :])[dore_mask]

        if 'd3dwor' in type:
            result['d3dwor'] = np.array(dense_data[3, :, :])

        if 'dvor' in type:
            result['dvor'] = np.array(dense_data[2, :, :])

        if 'd2dwor' in type:
            result['d2dwor'] = np.array(dense_data[5, :, :])

        if 'dde' in type:
            result['dde'] = np.array(dense_data[4, :, :])

        if 'ddee' in type:
            result['ddee'] = np.array(dense_data[4, :, :])
            dore_mask = result['ddee'] < epsilon
            result['ddee'][dore_mask] = np.array(dense_data[2, :, :])[dore_mask]

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

        # for key, value in channels_transformed_np.items():
        #     if key == 'gt_depth':
        #         continue

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
        target_data = None
        target_data = channels_transformed_np['gt_depth']
        #np.stack([channels_transformed_np['gt_depth'],channels_transformed_np['normal_x'],channels_transformed_np['normal_y'],channels_transformed_np['normal_z']])
        # target_data = self.append_tensor3d(target_data,channels_transformed_np['gt_depth'])
        # target_data = self.append_tensor3d(target_data, channels_transformed_np['normal_x'])
        # target_data = self.append_tensor3d(target_data, channels_transformed_np['normal_y'])
        # target_data = self.append_tensor3d(target_data, channels_transformed_np['normal_z'])

        target_depth_tensor = self.to_tensor(target_data).unsqueeze(0)


        #target_depth_tensor = depth_tensor.unsqueeze(0)
        #        if input_tensor.dim() == 2: #force to have a third dimension on the single channel input
#            input_tensor = input_tensor.unsqueeze(0)
#        if input_tensor.dim() < 2:
#            raise (RuntimeError("transform not defined"))
 #       depth_tensor = totensor(depth_np)
#        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, target_depth_tensor, channels_transformed_np['scale']

    def __len__(self):
        return len(self.imgs)

class SeqMyDataloaderExt(MyDataloaderExt):

    def __init__(self, root, type, sparsifier=None, modality='rgb',sequence_size=2,skip_step=10):
        super(SeqMyDataloaderExt,self).__init__(root,type,sparsifier,modality)

    def __getitem__(self, index):

