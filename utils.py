import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import cv2

cmap = plt.cm.viridis

def parse_command():
    model_names = ['resnet18', 'resnet34', 'resnet50','depthcompnet18','depthcompnet34','depthcompnet50','weightcompnet18','weightcompnet34','weightcompnet50']
    loss_names = ['l1', 'l2']
    data_names = ['nyudepthv2', 'kitti', 'visim']
    depth_weight_head_type_names = ['CBR','ResBlock1']
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
    output_directory = os.path.join('results',
        '{}.dw_head={}.samples={}.modality={}.arch={}.criterion={}.divider={}.lr={}.lrs={}.bs={}.pretrained={}'.
        format(args.data, args.depth_weight_head_type, args.num_samples, args.modality, \
            args.arch,  args.criterion, args.depth_divider, args.lr,args.lrs, args.batch_size, \
            'file' if args.pretrained else str(args.pretrained)))
    return output_directory


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


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


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    mask = np.logical_and(depth_input_cpu > 10e-5 ,  depth_target_cpu > 10e-5)

    d_min = min(np.min(depth_input_cpu[mask]), np.min(depth_target_cpu[mask]), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    hist = write_minmax(rgb.shape,d_min,d_max)
    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col,hist])

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
    img_merge.save(filename)
