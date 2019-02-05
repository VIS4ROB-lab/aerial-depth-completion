import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import csv 
import dataloaders.transforms as transforms
import math

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

totensor = transforms.ToTensor()

class MyDataloaderExt(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd','keypoint_original','keypoint_gt','keypoint_denoise','dense_original','dense_denoise'] # , 'g', 'gd'
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

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, input_depth, targe_depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, input_depth, targe_depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def pack_rgbd(self, rgb, depth):
        rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)
        return rgbd

    def h5_loader_original(self,index):
        raise (RuntimeError("todo"))
        path, target = self.imgs[index]
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f['rgb_image_data'])
        rgb = np.transpose(rgb, (1, 2, 0))
        dense_data = h5f['dense_image_data']
        depth = np.array(dense_data[0, :, :])
        mask_array = depth > 5000
        depth[mask_array] = 0
        return rgb, depth

    def h5_loader_slam_keypoints(self,index,type):
        path, target = self.imgs[index]
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f['rgb_image_data'])
        rgb = np.transpose(rgb, (1, 2, 0))
        dense_data = h5f['dense_image_data']
        depth = np.array(dense_data[0, :, :])
        mask_array = depth > 5000
        depth[mask_array] = 0
        data_2d = np.array(h5f['landmark_2d_data'])
        sparse_input = np.zeros_like(depth)
        for row in data_2d:
            xp = int(math.floor(row[1]))
            yp = int(math.floor(row[0]))
            if type == 'keypoint_gt':
                if(depth[xp,yp] > 0):
                    sparse_input[xp,yp] = depth[xp,yp]
            elif type == 'keypoint_original':
                if(row[2] > 0):
                    sparse_input[xp,yp] = row[2]
            elif type == 'keypoint_denoise':
                if(row[3] > 0):
                    sparse_input[xp,yp] = row[3]
            else:
                raise (RuntimeError("type input sparse not defined"))
        return rgb, sparse_input, depth

    def h5_loader_slam_dense(self,index,type):
        path, target = self.imgs[index]
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f['rgb_image_data'])
        rgb = np.transpose(rgb, (1, 2, 0))
        dense_data = h5f['dense_image_data']
        depth = np.array(dense_data[0, :, :])
        mask_array = depth > 5000
        depth[mask_array] = 0

        if type == 'dense_original':
            prior_depth_input = np.array(dense_data[1, :, :])
        elif type == 'dense_denoise':
            prior_depth_input = np.array(dense_data[4, :, :])
        else:
            raise (RuntimeError("type input sparse not defined"))
        return rgb, prior_depth_input, depth



    # def __getraw__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (rgb, depth) the raw data.
    #     """
    #     path, target = self.imgs[index]
    #     rgb, depth = self.loader(path)
    #
    #     return rgb, depth

    def __getitem__(self, index):

        input_np =[]

        if self.modality == 'rgb' or self.modality == 'rgbd' or self.modality == 'd':
            rgb, depth = self.h5_loader_original(index)
            if self.transform is not None:
                rgb_np, depth_np = self.transform(rgb, depth)
            else:
                raise(RuntimeError("transform not defined"))

            # color normalization
            # rgb_tensor = normalize_rgb(rgb_tensor)
            # rgb_np = normalize_np(rgb_np)

            if self.modality == 'rgb':
                input_np = rgb_np
            elif self.modality == 'rgbd':
                input_np = self.create_rgbd(rgb_np, depth_np)
            elif self.modality == 'd':
                input_np = self.create_sparse_depth(rgb_np, depth_np)
        elif self.modality == 'dense_original' or self.modality == 'dense_denoise' :
            rgb, sparse_input, depth = self.h5_loader_slam_dense(index, self.modality)
            if self.transform is not None:
                rgb_np, sparse_input_np, depth_np = self.transform(rgb, sparse_input, depth)
            else:
                raise (RuntimeError("transform not defined"))
            input_np = self.pack_rgbd(rgb_np, sparse_input_np)
        else:
            rgb, sparse_input, depth = self.h5_loader_slam_keypoints(index,self.modality)
            if self.transform is not None:
                rgb_np, sparse_input_np, depth_np = self.transform(rgb, sparse_input, depth)
            else:
                raise (RuntimeError("transform not defined"))
            input_np = self.pack_rgbd(rgb_np, sparse_input_np)

        input_tensor = totensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = totensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)
