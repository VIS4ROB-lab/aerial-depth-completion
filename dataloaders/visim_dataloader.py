import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader_ext import MyDataloaderExt

iheight, iwidth = 480, 752 # raw image size

class VISIMDataset(MyDataloaderExt):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(VISIMDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (228, 304)

    def train_transform(self, rgb, no_scale, attrib_list):
        if no_scale:
            s = 1.0
        else:
            s = np.random.uniform(1.0, 1.5) # random scaling

        attrib_np = []

        for dp in attrib_list:
            attrib_np.append(dp / s)

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
        attrib_result_np = []
        for dp in attrib_np:
            attrib_result_np.append(transform(dp))


        return rgb_np, attrib_result_np

    def val_transform(self, rgb, attrib_list):

        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255

        attrib_np = []

        for dp in attrib_list:
            attrib_np.append(transform(dp))

        return rgb_np, attrib_np
