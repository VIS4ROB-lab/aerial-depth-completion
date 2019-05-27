import os
import torch
import numpy as np
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo


def create_data_loaders(data_path, data_type='visim', loader_type='val', arch='', sparsifier_type='uar', num_samples=500, modality='rgb-fd', depth_divisor=1, max_depth=-1, max_gt_depth=-1, batch_size=8, workers=8):
    # Data loading code
    print("=> creating data loaders ...")

    #legacy compatibility with sparse-to-dense data folder
    subfolder = os.path.join( data_path, loader_type )
    # if os.path.exists(subfolder):
    #     data_path = subfolder

    if not os.path.exists(data_path):
        raise RuntimeError('Data source does not exit:{}'.format(data_path))

    loader = None
    dataset = None
    max_depth = max_depth if max_depth >= 0.0 else np.inf
    max_gt_depth = max_gt_depth if max_gt_depth >= 0.0 else np.inf

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None

    if sparsifier_type == UniformSampling.name: #uar
        sparsifier = UniformSampling(num_samples=num_samples, max_depth=max_depth)
    elif sparsifier_type == SimulatedStereo.name: #sim_stereo
        sparsifier = SimulatedStereo(num_samples=num_samples, max_depth=max_depth)

    if data_type == 'kitti':
        from dataloaders.kitti_loader import KittiDepth

        dataset = KittiDepth(data_path, split=loader_type, depth_divisor=depth_divisor)

    elif data_type == 'visim':
        from dataloaders.visim_dataloader import VISIMDataset

        dataset = VISIMDataset(data_path, type=loader_type,
                               modality=modality, sparsifier=sparsifier, depth_divider=depth_divisor, is_resnet= ('resnet' in arch), max_gt_depth=max_gt_depth)

    elif data_type == 'visim_seq':
        from dataloaders.visim_dataloader import VISIMSeqDataset
        dataset = VISIMSeqDataset(data_path, type=loader_type,
                                  modality=modality, sparsifier=sparsifier, depth_divider=depth_divisor, is_resnet= ('resnet' in arch), max_gt_depth=max_gt_depth)
    else:
        raise RuntimeError('data type not found.' +
                           'The dataset must be either of kitti, visim or visim_seq.')

    if loader_type == 'val':
        # set batch size to be 1 for validation
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
        print("=> Val loader:{}".format(len(dataset)))
    elif loader_type == 'train':
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
        print("=> Train loader:{}".format(len(dataset)))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return loader, dataset
