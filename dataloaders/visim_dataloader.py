import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader_ext import MyDataloaderExt

iheight, iwidth = 480, 752 # raw image size

class VISIMDataset(MyDataloaderExt):
    def __init__(self, root, type, sparsifier=None, modality='rgb',depth_divider=1.0):
        super(VISIMDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (228, 304) #(240, 320)#
        self.depth_divider = depth_divider

    def train_transform(self, attrib_list):

        s = np.random.uniform(1.0, 1.2) # random scaling

        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])

        attrib_np = dict()

        for key, value in attrib_list.items():
            attrib_np[key] = transform(value)
            if key in ['gt_depth','fd','kor','kde','kgt','dor','dde']:
                attrib_np[key] = attrib_np[key]  / self.depth_divider

        if 'rgb' in attrib_np:
            attrib_np['rgb'] = self.color_jitter(attrib_np['rgb'])  # random color jittering
            attrib_np['rgb'] = (np.asfarray(attrib_np['rgb'], dtype='float') / 255).transpose((2, 0, 1))#all channels need to have C x H x W

        return attrib_np

    def val_transform(self,  attrib_list):

        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])

        attrib_np = dict()

        for key, value in attrib_list.items():
            attrib_np[key] = transform(value)
            if key in ['gt_depth','fd','kor','kde','kgt','dor','dde']:
                attrib_np[key] = attrib_np[key]  / self.depth_divider

        if 'rgb' in attrib_np:
            attrib_np['rgb'] = (np.asfarray(attrib_np['rgb'], dtype='float') / 255).transpose((2, 0, 1))

        return attrib_np
