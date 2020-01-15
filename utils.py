import math
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import ndimage

epsilon= np.finfo(float).eps
cmap = plt.cm.viridis

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch, lr_init,lr_step,lr_min):
    """Sets the learning rate to the initial LR decayed by 10 every step epochs"""
    if lr_step < 1:
        lr = lr_init
    else:
        lr = lr_init * (0.1 ** (epoch // lr_step))

    if lr < lr_min:
        lr = lr_min

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_output_directory(args):
    if isinstance(args.pretrained, bool):
        pretrain_text= str(args.pretrained)
    else:
        head, tail = os.path.split(args.pretrained)
        pretrain_text=tail

    output_directory = os.path.join('results',
        '{}.dw_head={}.samples={}.modality={}.arch={}.criterion={}.divider={}.lr={}.lrs={}.bs={}.pretrained={}'.
        format(args.data, args.depth_weight_head_type, args.num_samples, args.modality, \
            args.arch,  args.criterion, args.depth_divider, args.lr,args.lrs, args.batch_size, \
            pretrain_text))
    return output_directory


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

confidence_color_map = plt.cm.jet #plt.cm.seismic gist_rainbow
def confidence_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * confidence_color_map(depth_relative)[:,:,:3] # H, W, C

confidence_thres_color_map = plt.cm.binary
def confidence_thres_depthmap(depth):
    return 255 * confidence_thres_color_map(depth)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    
    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred,normal_target=None,normal_pred=None,valid_mask=None,new_prediction=None):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    if normal_target is not None:
        normal_target_cpu = 127.5 * (np.transpose(np.squeeze(normal_target.cpu().numpy()), (1,2,0))+1)
    else:
        normal_target_cpu = np.zeros_like(rgb)

    if normal_pred is not None:
        normal_pred_cpu = 127.5 * (np.transpose(np.squeeze(normal_pred.cpu().numpy()), (1,2,0))+1)
    else:
        normal_pred_cpu = np.zeros_like(rgb)

    if valid_mask is not None:
        valid_mask_cpu = np.squeeze(valid_mask.cpu().numpy())
    else:
        valid_mask_cpu = np.zeros_like(depth_input_cpu)

    if new_prediction is not None:
        new_depth_pred_cpu = np.squeeze(new_prediction.cpu().numpy())
    else:
        new_depth_pred_cpu = np.zeros_like(depth_input_cpu)


    input_depth_mask  = depth_input_cpu  > 10e-5
    target_depth_mask = depth_target_cpu > 10e-5
    if input_depth_mask.sum() > 0:
        mask = np.logical_and( input_depth_mask,  target_depth_mask)
    else:
        mask = target_depth_mask

    d_min = min(np.min(depth_input_cpu[mask]), np.min(depth_target_cpu[target_depth_mask]), np.min(depth_pred_cpu[target_depth_mask]))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))

    #d_input_min = np.min(depth_input_cpu[mask])
    ##d_input_max = np.max(depth_input_cpu[mask])
    #d_pred_min = np.min(depth_pred_cpu[mask])
    #d_pred_max = np.max(depth_pred_cpu[mask])

    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    new_depth_pred_col = colored_depthmap(new_depth_pred_cpu, d_min, d_max)
    hist = write_minmax(rgb.shape,d_min,d_max)

    abs_diff = np.absolute((depth_pred_cpu - depth_target_cpu))
    absrel = np.zeros_like(abs_diff)
    absrel[target_depth_mask] = abs_diff[target_depth_mask]/ depth_target_cpu[target_depth_mask]
    diff_col_abs = colored_depthmap(abs_diff, 0, 5)

    diff_col_rel = colored_depthmap(absrel, 0, 0.1)
    diff_col_rel01 = colored_depthmap(absrel, 0, 0.05)
    #diff_col_rel01_pred = confidence_depthmap(valid_mask_cpu, 0, 1)
    diff_col_rel01_pred = confidence_depthmap(valid_mask_cpu, 0, 1)
    #threshold_indices = valid_mask_cpu < 0.5
    #valid_thres = np.zeros_like(valid_mask_cpu)
    #valid_thres[threshold_indices] = 1
    #diff_col_rel01_pred_thres =confidence_thres_depthmap(valid_thres)

    img_merge = np.hstack([rgb, depth_input_col, normal_target_cpu,normal_pred_cpu, depth_target_col, depth_pred_col,hist,diff_col_abs,diff_col_rel,diff_col_rel01,diff_col_rel01_pred,new_depth_pred_col])

    return img_merge

jet_color_map = plt.cm.jet #plt.cm.seismic gist_rainbow

def colored_depthmap2(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min+10e-8)
    return 255 * jet_color_map(depth_relative)[:,:,:3] # H, W, C



def merge_into_row_with_gt2(rgb, input_depth,input_conf, in_gt_depth , out_depth1, out_conf1=None, out_depth2=None):
    rgb = 255 * np.transpose(np.squeeze(rgb.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(input_depth.cpu().numpy())
    depth_conf_cpu = np.squeeze(input_conf.cpu().numpy())
    depth_target_cpu = np.squeeze(in_gt_depth.cpu().numpy())
    depth_pred_cpu = np.squeeze(out_depth1.detach().cpu().numpy())

    if out_conf1 is not None:
        out_conf_cpu = np.squeeze(out_conf1.detach().cpu().numpy())
    else:
        out_conf_cpu = np.zeros_like(depth_input_cpu)

    if out_depth2 is not None:
        out_depth2_cpu = np.squeeze(out_depth2.detach().cpu().numpy())
    else:
        out_depth2_cpu = np.zeros_like(depth_input_cpu)

    target_depth_mask = depth_target_cpu > 10e-5
    sparse_depth_mask = depth_input_cpu < 10e-5


#depth colormap
    d_min = np.min(depth_target_cpu[target_depth_mask])
    d_max = np.max(depth_target_cpu)

    depth_input_col = colored_depthmap2(depth_input_cpu, d_min, d_max)
    depth_input_col[sparse_depth_mask,:] = 0
    depth_target_col = colored_depthmap2(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap2(depth_pred_cpu, d_min, d_max)
    depth2_pred_col = colored_depthmap2(out_depth2_cpu, d_min, d_max)
    depth_target_col[~target_depth_mask, :] = 0

#conf_colormap
    c_min = np.min(out_conf_cpu[target_depth_mask])
    c_max = np.max(out_conf_cpu[target_depth_mask])

    out_conf_col = colored_depthmap2(out_conf_cpu, c_min, c_max)
    depth_conf_col = colored_depthmap2(depth_conf_cpu)
    depth_conf_col[sparse_depth_mask,:] = 0


    hist = write_minmax(rgb.shape,d_min,d_max,c_min,c_max)


    abs_diff = np.absolute((depth_pred_cpu - depth_target_cpu))
    absrel = np.zeros_like(abs_diff)
    absrel[target_depth_mask] = abs_diff[target_depth_mask]/ depth_target_cpu[target_depth_mask]
    diff_col_abs = colored_depthmap2(abs_diff, 0, 5)
    diff_col_abs[~target_depth_mask,:]=0
    diff_col_rel = colored_depthmap2(absrel, 0, 0.1)
    diff_col_rel[~target_depth_mask, :] = 0

    img_merge = np.hstack([rgb, depth_input_col, depth_conf_col, depth_target_col, depth_pred_col,
                           depth2_pred_col, out_conf_col, diff_col_abs, diff_col_rel, hist])

    return img_merge

def write_minmax(size_image,dmin,dmax,cmin,cmax):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.text(0.0, 0.0, "dmin:{0:.4f}\ndmax:{1:.4f}\ncmin:{2:.4f}\ncmax:{3:.4f}".format(dmin,dmax,cmin,cmax), fontsize=45)
    ax.axis('off')

    canvas.draw()  # draw the canvas, cache the renderer
    ncols, nrows = fig.canvas.get_width_height()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape( nrows, ncols, 3)
    res = cv2.resize(image, dsize=(size_image[1],size_image[0]), interpolation=cv2.INTER_CUBIC)
    return res


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)#,optimize=False,compress_level=0

def depth_to_normal_map(depth,use_sobel=True,dtype='uint8'):

    if use_sobel:
        zx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5,scale=1/128.) # or ksize=3 and scale=1/8.
        zy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5,scale=1/128.) # or ksize=3 and scale=1/8.
        nx, ny, nz = zx, -zy, np.ones_like(depth)
    else:
        zy, zx = np.gradient(depth.astype(np.float32))
        nx,ny,nz = zx,  -zy, np.ones_like(depth) # x left-right, y down-up

    normal = np.dstack((nx,ny,nz))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    if dtype == 'uint8':
        normal[:, :, :] += 1
        normal[:, :, :] *= 127.5
        #normal[:, :, :] /= 2
        #normal[:, :, :] *= 255

    return normal.astype(dtype)

def calc_from_sparse_input( in_sparse_map, voronoi=True, edt=True):
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

    return res_voronoi, res_edt
