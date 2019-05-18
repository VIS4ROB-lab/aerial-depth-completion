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

def load_files_list(path, base_filter):
    images = []    
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        folder = os.path.dirname(path)
        for row in reader:            
            path = os.path.join(folder, row[1])
            if base_filter is None or base_filter in path:
                item = (path,row[0])
                images.append(item)
    return images


def find_classes(dir, base_filter=None):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and (base_filter is None or base_filter in d)]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in class_to_idx:
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def load_class_dataset(dir,class_):
    images = []
    dir = os.path.expanduser(dir)
    target = class_
    d = os.path.join(dir, target)
    assert( os.path.isdir(d)), 'path is not a folder'

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images
#
# def load_extra_datasets(data_folder,type, image_list):
#
#     extra_folder = os.path.join(data_folder,'extra')
#     extra_paths = [None]* len(image_list)
#     extra_count = 0
#     for i in range(len(image_list)):
#         curr_path,_ = image_list[i]
#         curr_relpath_in_extra = os.path.relpath(curr_path,os.path.join(data_folder,type))
#         curr_abspath_in_extra = os.path.join(extra_folder,curr_relpath_in_extra)
#
#         if os.path.exists(curr_abspath_in_extra):
#             extra_count = extra_count + 1
#             extra_paths[i] = curr_abspath_in_extra
#
#     return extra_paths, extra_count

def load_class_extras(data_folder, type, image_list):

    extra_folder = os.path.join(data_folder,'extra')
    extra_paths = []
    img_paths = []

    for i in range(len(image_list)):
        curr_path = image_list[i]
        curr_relpath_in_extra = os.path.relpath(curr_path,os.path.join(data_folder,type))
        curr_abspath_in_extra = os.path.join(extra_folder,curr_relpath_in_extra)

        if os.path.exists(curr_abspath_in_extra):
            extra_paths.append(curr_abspath_in_extra)
            img_paths.append(curr_path)

    return extra_paths, img_paths


def rgb2grayscale(rgb):
    return rgb[0,:,:] * 0.2989 + rgb[1,:,:] * 0.587 + rgb[2,:,:] * 0.114

class Modality():
    #modality_names = ['rgb', 'grey', 'fd', 'kor', 'kgt', 'kw', 'kde', 'dor', 'dde', 'dvor', 'd2dwor', 'd3dwde','d3dwor','wkde','wdde']

    depth_channels_names = ['fd', 'kor', 'kde', 'kgt', 'dor', 'dde', 'dvor', 'dvgt', 'dvde','dore','ddee']
    metric_weight_names = ['d3dwde','d3dwor','wkde','wdde']
    image_size_weight_names = ['d2dwde','d2dwgt','d2dwor']
    weight_names = image_size_weight_names + ['kw'] + metric_weight_names
    need_divider = depth_channels_names + ['gt_depth'] + metric_weight_names
    no_transform = ['t_wc','scale']
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

    def __init__(self, root, type, sparsifier=None,max_gt_depth=math.inf, modality='rgb'):
        
        imgs = []

        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise RuntimeError('invalid type of dataset')

        dataset_folder = os.path.join(root, type)
        classes, class_to_idx = find_classes(dataset_folder)
        imgs = make_dataset(dataset_folder, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx


        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.extra_ds = None
        #self.classes = classes
        #self.class_to_idx = class_to_idx

        self.sparsifier = sparsifier
        self.modality = Modality(modality)
        self.max_gt_depth = max_gt_depth



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

#pose = none | gt | slam
    def h5_loader_general(self,img_path,extra_path,type,pose='none'):
        result = dict()
        #path, target = self.imgs[index]
        h5f = h5py.File(img_path, "r")
        h5fextra = None
        if extra_path is not None:
            h5fextra = h5py.File(extra_path, "r")

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
            depth = np.array(h5f['depth'])
            if not math.isinf(self.max_gt_depth):
                mask_max = depth >  self.max_gt_depth
                depth[mask_max] = 0
            result['gt_depth'] = depth

        if pose == 'gt':
            if h5fextra is not None:
                result['t_wc'] = np.array(h5fextra['gt_twc_data'])
            else:
                if 'gt_twc_data' not in h5f:
                    return None
                result['t_wc'] = np.array(h5f['gt_twc_data'])
                assert result['t_wc'].shape == (4, 4), 'file {} - the t_wc is not 4x4'.format(path)

        if pose == 'slam':
            if h5fextra is not None:
                result['t_wc'] = np.array(h5fextra['slam_twc_data'])
            else:
                if 'slam_twc_data' not in h5f:
                    return None
                result['t_wc'] = np.array(h5f['slam_twc_data'])
                assert result['t_wc'].shape == (4, 4), 'file {} - the t_wc is not 4x4'.format(path)

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
            if h5fextra is not None:
                data_2d = np.array(h5fextra['landmark_2d_data'])
            else:
                if 'landmark_2d_data' not in h5f:
                    return None
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
        image_path,_ = self.imgs[index]
        channels_np = self.h5_loader_general(image_path,None, self.modality)

        if channels_np is None:
            return None,None,None

        if self.transform is not None:
            channels_transformed_np = self.transform(channels_np)
        else:
            raise (RuntimeError("transform not defined"))

        # for key, value in channels_transformed_np.items():
        #     if key == 'gt_depth':
        #         continue

        target_data = None
        target_data = channels_transformed_np['gt_depth']

        num_image_channel,image_channel = self.modality.get_input_image_channel()
        if num_image_channel > 0:
            input_np = self.append_tensor3d(input_np,channels_transformed_np[image_channel])

        num_depth_channel, depth_channel = self.modality.get_input_depth_channel()
        if num_depth_channel > 0:
            input_np = self.append_tensor3d(input_np, channels_transformed_np[depth_channel])

        num_weight_channel, weight_channel = self.modality.get_input_weight_channel()
        if num_weight_channel > 0:
            input_np = self.append_tensor3d(input_np, channels_transformed_np[weight_channel])
        else:
            confidence = np.zeros_like(input_np[0,:,:])
            valid_mask = ((channels_transformed_np['gt_depth'] > 0))
            confidence[valid_mask] = 1.0
            input_np = self.append_tensor3d(input_np, confidence)

        input_tensor = self.to_tensor(input_np)


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

    def __init__(self, root, type, sparsifier=None,max_gt_depth=math.inf, modality='rgb',sequence_size=2,skip_step=5):
     #   super(SeqMyDataloaderExt,self).__init__(root,type,sparsifier,max_gt_depth,modality,base_filter='ds')
        #self.extra_ds,num_extras = load_extra_datasets(root,type,self.imgs)
        #print ("loaded new {} extras".format(num_extras))
        self.skip_step = skip_step
        self.sequence_size = sequence_size
        self.begging_offset = (sequence_size-1)*skip_step

        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise RuntimeError('invalid type of dataset')

        dataset_folder = os.path.join(root, type)

        general_img_index = []

        classes, class_to_idx = find_classes(dataset_folder, 'ds')
        general_class_data = [ None ] * len(classes)
        for i_class,curr_class in enumerate(classes):
            class_images = load_class_dataset(dataset_folder,curr_class)
            class_extras = None
            if 'dsx' in curr_class:
                class_extras, class_images = load_class_extras(dataset_folder,curr_class,class_images)
            general_class_data[i_class] = dict(name=curr_class, images=class_images, extras=class_extras)

            for i_img in range(self.begging_offset,len(class_images)):
                general_img_index.append((i_class,i_img))

        #imgs = make_dataset(dataset_folder, class_to_idx)
        #self.classes = classes
        #self.class_to_idx = class_to_idx

        assert len(general_img_index) > 0, "Found 0 images compatible with sequencing in subfolders of: " + root + "\n"
        print("Found {} images compatible with sequencing in {} folder.".format(len(general_img_index), type))

        self.root = root
        self.general_img_index = general_img_index
        self.general_class_data = general_class_data

        self.sparsifier = sparsifier
        self.modality = Modality(modality)
        self.max_gt_depth = max_gt_depth

    def __len__(self):
        return len(self.general_img_index)

    def load_one_sample(self, class_idx, img_idx,sequence_scale):
        input_np = None
        class_entry = self.general_class_data[class_idx]
        img_path = class_entry['images'][img_idx]
        extra_path = (class_entry['extras'][img_idx] if class_entry['extras'] is not None else None )
        channels_np = self.h5_loader_general(img_path, extra_path, self.modality,pose='gt')
        channels_np['scale'] = sequence_scale

        assert (channels_np is not None),"error in loading {} , {}".format(img_path,extra_path)

        if self.transform is not None:
            channels_transformed_np = self.transform(channels_np)
        else:
            raise (RuntimeError("transform not defined"))

        num_image_channel, image_channel = self.modality.get_input_image_channel()
        if num_image_channel > 0:
            input_np = self.append_tensor3d(input_np, channels_transformed_np[image_channel])

        num_depth_channel, depth_channel = self.modality.get_input_depth_channel()
        if num_depth_channel > 0:
            input_np = self.append_tensor3d(input_np, channels_transformed_np[depth_channel])

        num_weight_channel, weight_channel = self.modality.get_input_weight_channel()
        if num_weight_channel > 0:
            input_np = self.append_tensor3d(input_np, channels_transformed_np[weight_channel])

        input_tensor = self.to_tensor(input_np)

        target_depth_tensor = self.to_tensor(channels_transformed_np['gt_depth']).unsqueeze(0)

        return input_tensor, target_depth_tensor, channels_transformed_np['scale'], channels_transformed_np['t_wc']

    # def find_near_frames(self, index):
    #     _, sequence = self.imgs[index]
    #     #search after
    #     result = [index]
    #     for i in range(1,self.sequence_size):
    #         next = index + i*self.skip_step
    #         if next < len(self.imgs):
    #             _,next_sequence =self.imgs[next]
    #             if sequence == next_sequence:
    #                 result.append(next)
    #             else:
    #                 break
    #
    #     #search before
    #     for i in range(1,self.sequence_size - len(result)):
    #         next = index - i*self.skip_step
    #         if next > 0:
    #             _,next_sequence =self.imgs[next]
    #             if sequence == next_sequence:
    #                 result.append(next)
    #             else:
    #                 break
    #
    #     result.sort()
    #     return result




    def __getitem__(self, index):

        idx_class,idx_img = self.general_img_index[index]

        result_input_tensor = []
        result_target_tensor = []
        result_scale = -1
        result_transforms = []
        for frame in range(self.sequence_size):
            curr_img_idx = idx_img - (frame * self.skip_step)
            curr_input,curr_target,curr_scale,transform = self.load_one_sample(idx_class,curr_img_idx)

            if curr_input is None:
                return None, None, None, None

            result_input_tensor.append(curr_input)
            result_target_tensor.append(curr_target)
            result_scales.append(curr_scale)
            result_transforms.append(transform)

        return result_input_tensor,result_target_tensor,result_scales,result_transforms