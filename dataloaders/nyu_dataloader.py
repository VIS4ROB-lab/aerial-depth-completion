import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size

class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb', arch='resnet18',depth_divider=1.0):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)

        self.depth_divider = depth_divider
        self.arch = arch

        if 'depthcompnet' in self.arch:
            self.output_size = (240, 320)
        elif 'resnet' in self.arch:
            self.output_size = (228, 304)
        else:
            raise (RuntimeError("{} is an unknown arch - visim-dataloader".format(self.arch)))

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / (s*self.depth_divider)
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth/ (self.depth_divider)
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np
