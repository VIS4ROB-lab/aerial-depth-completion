import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import csv 
import dataloaders.transforms as transforms

IMG_EXTENSIONS = ['.h5',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def h5_loader_ext(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb_image_data'])
    rgb = np.transpose(rgb, (1, 2, 0))
    dense_data = h5f['dense_image_data']
    depth = np.array(dense_data[0,:,:])
    return rgb, depth
    
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
    modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader_ext):
        
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
        
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, depth):
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

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
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

        input_tensor = totensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = totensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)
