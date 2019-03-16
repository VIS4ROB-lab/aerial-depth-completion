import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import ndimage
from PIL import Image
import cv2

epsilon= np.finfo(float).eps
cmap = plt.cm.viridis

def parse_command():
    model_names = ['resnet18', 'resnet34', 'resnet50','depthcompnet18','depthcompnet34','depthcompnet50','sdepthcompnet18','vdepthcompnet18','vdepthcompnet34','vdepthcompnet50','weightcompnet18','weightcompnet34','weightcompnet50']
    loss_names = ['l1', 'l2','l2gn','l2nv','l1smooth','wl1smooth']
    data_names = ['nyudepthv2', 'kitti', 'visim']
    depth_weight_head_type_names = ['CBR','ResBlock1','JOIN']
    from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
    sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo]]
    from models import Decoder
    decoder_names = Decoder.names
    from dataloaders.dataloader_ext import Modality


    import argparse
    parser = argparse.ArgumentParser(description='Aerial Depth Completion')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--depth-weight-head-type', '-dwht', metavar='TYPE', default='CBR', choices=depth_weight_head_type_names,
                        help='head architecture: ' + ' | '.join(depth_weight_head_type_names) + ' (default: CBR)')

    parser.add_argument('--data', metavar='DATA', default='visim',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    parser.add_argument('--data-path', default='data', type=str, metavar='PATH',
                        help='path to data folder')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', type=str,
                        help='modality: ' + ' | '.join(Modality.modality_names) + ' (default: rgb-fd)')
    parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                        help='number of sparse depth samples (default: 0)')
    parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--depth-divider', default=1.0, type=float, metavar='D',
                        help='Normalization factor (default: 1.0 [m])')
    parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names,
                        help='sparsifier: ' + ' | '.join(sparsifier_names) + ' (default: ' + UniformSampling.name + ')')
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv3', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: deconv2)')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,dest='lr',
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('-lrs', '--learning-rate-step', default=5, type=int, metavar='LRS',dest='lrs',
                        help='number of epochs between reduce the learning rate by 10 (default: 5)')
    parser.add_argument('-lrm', '--learning-rate-min', default=0.00001, type=float, dest='lrm',
                        metavar='LRM', help='minimum learning rate (default 0.00001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='resnet', type=str, metavar='PATH',
                        help='path to pretraining checkpoint (default: buildin resnet)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')
    #parser.set_defaults(pretrained=True)
    args = parser.parse_args()


    if args.pretrained == 'resnet':
        args.pretrained = True
    elif args.pretrained != '':
        assert os.path.isfile(args.pretrained)
    else:
        args.pretrained = False

    if not Modality.validate_static(args.modality):
        print("input modality with problem")
        exit(0)

    # if args.modality == 'rgb' and args.max_depth != 0.0:
    #     print("max depth is forced to be 0.0 when input modality is rgb/rgbd")
    #     args.max_depth = 0.0
    return args

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

confidence_color_map = plt.cm.seismic
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


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred,normal_target=None,normal_pred=None,valid_mask=None):
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


    mask = np.logical_and(depth_input_cpu > 10e-5 ,  depth_target_cpu > 10e-5)

    d_min = min(np.min(depth_input_cpu[mask]), np.min(depth_target_cpu[mask]), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    hist = write_minmax(rgb.shape,d_min,d_max)

    abs_diff = np.absolute((depth_pred_cpu - depth_target_cpu))
    absrel = abs_diff/ depth_target_cpu
    diff_col_abs = colored_depthmap(abs_diff, 0, 5)

    diff_col_rel = colored_depthmap(absrel, 0, 1)
    diff_col_rel01 = colored_depthmap(absrel, 0, 0.1)
    diff_col_rel01_pred = confidence_depthmap(valid_mask_cpu, 0, 1)
    diff_col_rel01_pred = confidence_depthmap(valid_mask_cpu, 0, 1)
    threshold_indices = valid_mask_cpu < 0.5
    valid_thres = np.zeros_like(valid_mask_cpu)
    valid_thres[threshold_indices] = 1
    diff_col_rel01_pred_thres =confidence_thres_depthmap(valid_thres)

    img_merge = np.hstack([rgb, depth_input_col, normal_target_cpu,normal_pred_cpu, depth_target_col, depth_pred_col,hist,diff_col_abs,diff_col_rel,diff_col_rel01,diff_col_rel01_pred_thres,diff_col_rel01_pred])

    return img_merge

def write_minmax(size_image,dmin,dmax):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.text(0.0, 0.0, "min:{0:.4f}\nmax:{1:.4f}".format(dmin,dmax), fontsize=45)
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
