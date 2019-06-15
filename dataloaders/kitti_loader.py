import glob
import os
import os.path
from random import choice

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from dataloaders import transforms_kitti as transforms

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

def get_paths_and_transform(split, args):
    root_d = os.path.join(args.data_folder, 'kitti_depth')
    root_rgb = os.path.join(args.data_folder, 'kitti_rgb')
    assert (args.use_d or args.use_rgb or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        glob_gt = "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
        pattern_d = ("groundtruth","velodyne_raw")
        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
            return pnew
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_gt = "val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"
            pattern_d = ("groundtruth","velodyne_raw")
            def get_rgb_paths(p):
                ps = p.split('/')
                pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                return pnew
        elif args.val == "select":
            transform = no_transform
            glob_gt = "val_selection_cropped/groundtruth_depth/*.png"
            pattern_d = ("groundtruth_depth","velodyne_raw")
            def get_rgb_paths(p):
                return p.replace("groundtruth_depth","image")
    else:
        raise ValueError("Unrecognized split "+str(split))

    if glob_gt is not None:
        glob_gt = os.path.join(root_d,glob_gt)
        paths_gt = sorted(glob.glob(glob_gt))
        paths_d = [p.replace(pattern_d[0],pattern_d[1]) for p in paths_gt]
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else: # test and only has d or rgb
        raise ValueError("Unrecognized glob_gt ")

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise(RuntimeError("Found 0 images in data folders"))
    if len(paths_d) == 0 and args.use_d:
        raise(RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise(RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise(RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise(RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb":paths_rgb, "d":paths_d, "gt":paths_gt}
    return paths, transform

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    #rgb_png = np.asarray( img_file, dtype="float32" ).copy() / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.asarray(img_file, dtype='uint8').copy() # in the range [0,255]
    img_file.close()
    return rgb_png

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth,-1)
    return depth

oheight, owidth = 352, 1216

def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth

def train_transform(rgb, sparse, target, rgb_near, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
        if rgb_near is not None:
            rgb_near = transform_rgb(rgb_near);
    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target, rgb_near

def val_transform(rgb, sparse, target, rgb_near, args):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if rgb_near is not None:
        rgb_near = transform(rgb_near);
    return rgb, sparse, target, rgb_near

def no_transform(rgb, sparse, target, rgb_near, args):
    return rgb, sparse, target, rgb_near

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img,-1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img

def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 3
    candidates = [i-max_frame_diff for i in range(max_frame_diff*2+1) if i-max_frame_diff!=0]
    while True:
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number+random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path_rgb_tgt)

    return rgb_read(path_near)

class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self,data_path, split,depth_divisor):
        self.data_folder = data_path
        self.use_rgb = True
        self.use_g = False
        self.use_d = True
        self.use_pose = False
        self.jitter = 0.1
        self.val = 'full'
        self.depth_divisor = depth_divisor
        self.split = split
        paths, transform = get_paths_and_transform(split, self)
        self.paths = paths
        self.transform = transform
        self.K = None
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.use_rgb or self.use_g)) else None
        sparse = depth_read(self.paths['d'][index]) if \
            (self.paths['d'][index] is not None and self.use_d) else None
        target = depth_read(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        # rgb_near = get_rgb_near(self.paths['rgb'][index], self.args) if \
        #     self.split == 'train' and self.args.use_pose else None
        return rgb, sparse, target, None

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
        rgb, sparse, target, rgb_near = self.__getraw__(index)
        rgb, sparse, target, rgb_near = self.transform(rgb,sparse, target, rgb_near, self)

        if self.depth_divisor == 0:
            max_depth = max(sparse.max(),1.0)
            scale = 10.0 / max_depth  # 10 is arbitrary. the network only converge in a especific range
        else:
            assert self.depth_divisor > 0 , 'divisor is negative'
            scale = 1.0 / self.depth_divisor


        input_np = np.append(rgb/255.0, sparse*scale, axis=2)
        input_np = input_np.transpose((2, 0, 1))
        confidence = np.zeros_like(input_np[0, :, :])
        valid_mask = ((input_np[3, :, :] > 0))
        confidence[valid_mask] = 1.0

        input_np = self.append_tensor3d(input_np, confidence)
        target_np = target.transpose((2, 0, 1))*scale

        input_ts = torch.from_numpy(input_np.copy()).float()
        target_ts = torch.from_numpy(target_np.copy()).float()
        iscale = 1/scale

        return input_ts, target_ts, iscale

    def __len__(self):
        return len(self.paths['gt'])


