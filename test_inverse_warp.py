
import h5py
import torch
import numpy as np
import inverse_warp as iw
import matplotlib.pyplot as plt


epsilon= np.finfo(float).eps

def inverse_pose(t_ab):
    # rot33 = t_ab[0:3,0:3]
    # t= t_ab[0:3, 3]
    rot33,t = decompose(t_ab)
    rot33_t = np.transpose(rot33)
    t_t = -(rot33_t.dot(t))
    # t_ba = np.identity(4,dtype='float64')
    # t_ba[0:3,0:3] = rot33_t
    # t_ba[0:3, 3] = t_t

    return compose(rot33_t,t_t) #t_ba #(rot33_t,t_t)

def compose(r_mat33,t_vec3):
    t_ba = np.identity(4, dtype='float32')
    t_ba[0:3, 0:3] = r_mat33
    t_ba[0:3, 3] = t_vec3
    return t_ba

def decompose(transform44):
    r_mat33 = transform44[0:3, 0:3]
    t_vec3 = transform44[0:3, 3]
    return r_mat33,t_vec3


def rgb2grayscale(rgb):
    return rgb[0,:,:] * 0.2989 + rgb[1,:,:] * 0.587 + rgb[2,:,:] * 0.114

np.set_printoptions(suppress=True)

im1_file = '/media/lucas/lucas-ds2-1tb/dataset_big_v8/ds-monteriggioni/239930000001.h5'
h5f1 = h5py.File(im1_file, "r")

im2_file = '/media/lucas/lucas-ds2-1tb/dataset_big_v8/ds-monteriggioni/240530000001.h5'
h5f2 = h5py.File(im2_file, "r")

t_wc1 = np.array(h5f1['gt_twc_data'])
t_wc2 = np.array(h5f2['gt_twc_data'])
t_c1w = inverse_pose(t_wc1)
t_c1c2 = t_c1w @ t_wc2

dense_data = h5f2['dense_image_data']
depth_c2 = np.array(dense_data[0, :, :])



rot,trans = decompose(t_c1c2)
r_mat = torch.from_numpy(rot.copy()).float()
t_vec = torch.from_numpy(trans.copy()).float()
rgb_near_np = np.array(h5f1['rgb_image_data'],dtype='float32')/80
rgb_near=torch.from_numpy(rgb_near_np).unsqueeze(0)

depth_curr = torch.from_numpy(depth_c2.copy()).unsqueeze(0).unsqueeze(0)
intrinsics = iw.Intrinsics(752,480,455,455,0,0)
result = iw.homography_from(rgb_near,depth_curr,r_mat,t_vec,intrinsics)
rgb_curr=np.array(h5f2['rgb_image_data'],dtype='float32')/80
img = result[0,:,:,:].numpy()
img[0,:,:] = rgb2grayscale(img)
img[1,:,:] = rgb2grayscale(rgb_curr)
img[2,:,:] = 0
imgplot = plt.imshow(img.transpose([1,2,0]))
plt.show()