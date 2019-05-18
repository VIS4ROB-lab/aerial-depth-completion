import os
import sys
import csv
import numpy as np
import math
import torch
import torch.backends.cudnn as cudnn

import torch.optim
from torchsummary import summary

import guided_ms_net

cudnn.benchmark = True

from model_zoo.s2d_resnet import S2DResNet
import model_zoo.model_conf as mc
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils
import argparse

# args = utils.parse_command()
# print(args)
# g_modality = Modality(args.modality)

# fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
#                 'delta1', 'delta2', 'delta3',
#                 'data_time', 'gpu_time','loss0','loss1','loss2']
# best_result = Result()
# best_result.set_to_worst()

class ConfidenceDepthFrameworkFactory():
    def __init__(self):
        a =1

    def create_dc_model(self,model_arch,pretrained_args,input_type,output_type):

        if pretrained_args == 'resnet':
            use_resnet_pretrained = True
        else:
            use_resnet_pretrained = False

        if model_arch == 'resnet18':
            #upproj is the best and default in the sparse-to-dense icra 2018 paper
            model = S2DResNet(layers=18, decoder='upproj', in_channels=len(input_type), out_channels=len(output_type), pretrained=use_resnet_pretrained)
        elif model_arch == 'nconv-ms':
            assert output_type == 'rgbdc', 'nconv-ms only accept rgbdc input'
            #upproj is the best and default in the sparse-to-dense icra 2018 paper
            model = guided_ms_net.NconvMS()

        else:
            raise RuntimeError('Model: {} not found.'.format(model_arch))

        return model

    def create_conf_model(self,model_arch,pretrained_args,dc_model):

        in_channels = dc_model.out_feature_channels
        out_channels = 1

        if model_arch == 'cbr3-c1':
            model = mc.CBR3C1Confidence(in_channels=in_channels)
        elif model_arch == 'forward':
            model = mc.ForwardConfidence(in_channels=in_channels)
        else:
            raise RuntimeError('Dataset not found.' +
                               'The dataset must be either of nyudepthv2 or kitti.')

        return model

    def create_loss(self, criterion, dual=False, weight1=0):

        if criterion == 'l2':
            loss = mc.MaskedMSELoss()
        elif criterion == 'l1':
            loss = mc.MaskedL1Loss()
        elif criterion == 'il1':
            loss = mc.InvertedMaskedL1Loss()
        elif criterion == 'absrel':
            loss = mc.MaskedAbsRelLoss()
        if dual:
            loss = mc.DualLoss(loss, loss, weight1)
        return loss

    def create_model(self, input_type, dc_arch, dc_weights, conf_arch
                     , conf_weights, lossdc_arch, lossdc_weights):

        model_dc = create_dc_model(args.dcnet_arch, args.dcnet_pretrained, args.dcnet_modality,output_type,output_size)

        if args.training_mode == 'dc1':
            model = model_dc


        if opt_parameters is None:
            opt_parameters = model.parameters()

        return model, opt_parameters




def create_command_parser():

    model_names = ['resnet18', 'udepthcompnet18','erfdepthcompnet','gms_depthcompnet','ged_depthcompnet']
    model_input_type = ['d','dw','c','cd','cdw']
    #image_type_source = ['g', 'rgb','undefined']
    sparse_depth_source = ['kgt','kor','kde','fd','undefined']
    sparse_conf_source = ['bin', 'kw','undefined']
    training_mode = ['dc1','dc1-ln0','dc1-ln1', 'dc0-cf1-ln0', 'dc1-cf1-ln1', 'dc1-cf1-ln0']
    confnet_exclusive_names = ['cbr3-c1', 'cbr3-cbr1-c1','cbr5-cbr3-cbr1-c1']
    confnet_names = confnet_exclusive_names + ['join','none']

    confnet_input_type = ['c','cd','cw','cdw','cdwr', 'cdwrl','cdr', 'cdrl', 'clr','lr'] #'c','cd','cw','cdw','cdwr', 'cdwrl','cdr', 'cdrl', 'clr'

    loss_names = ['l1', 'l2']
    data_types = ['nyuv2', 'visim', 'visim_nyuv2', 'kitti' ]
    data_scale = ['per_frame', 'global', 'none']

    opt_names = ['sgd', 'adam']
    from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
    sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo]]
    from models import Decoder
    decoder_names = Decoder.names

    parser = argparse.ArgumentParser(description='Confidence Depth Completion')

    # training
    parser.add_argument('--training-mode', metavar='ARCH', default='dc1', choices=training_mode,
                        help='training_mode: ' + ' | '.join(training_mode) + ' (default: dc1)')

    #dcnet
    parser.add_argument('--dcnet_arch', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')

    parser.add_argument('--dcnet_pretrained', default='resnet', type=str, metavar='PATH',
                        help='path to pretraining checkpoint (default: buildin resnet)')

    parser.add_argument('--dcnet_modality', metavar='MODALITY', default='cd', choices=model_input_type, type=str,
                        help='modality: ' + ' | '.join(model_input_type) + ' (default: cd)')

    #confnet
    parser.add_argument('--confnet_arch', metavar='ARCH', default='cbr3-c1', choices=confnet_names,
                        help='model architecture: ' + ' | '.join(confnet_names) + ' (default: cbr3-c1)')

    parser.add_argument('--confnet_pretrained', default='none', type=str, metavar='PATH',
                        help='path to pretraining checkpoint (default: none)')

    parser.add_argument('--confnet_modality', metavar='MODALITY', default='cdrl', choices=confnet_input_type, type=str,
                        help='modality: ' + ' | '.join(confnet_input_type) + ' (default: cdrl)')

    #lossnet
    parser.add_argument('--lossnet_arch', metavar='ARCH', default='ged_depthcompnet', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: ged_depthcompnet)')

    parser.add_argument('--lossnet_pretrained', default='none', type=str, metavar='PATH',
                        help='path to pretraining checkpoint (default: none)')

    # only useful for resnet
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv3', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: deconv2)')

    #input data
    parser.add_argument('--data-type', metavar='DATA', default='visim',
                        choices=data_types,
                        help='dataset: ' + ' | '.join(data_types) + ' (default: visim)')

    parser.add_argument('--data-path', default='data', type=str, metavar='PATH',
                        help='path to data folder')

    # parser.add_argument('--image-type', metavar='DATA', default='rgb',
    #                     choices=image_type_source,
    #                     help='dataset: ' + ' | '.join(image_type_source) + ' (default: rgb)')

    parser.add_argument('--sparse_depth-type', metavar='TYPE', default='rgb',
                        choices=sparse_depth_source,
                        help='dataset: ' + ' | '.join(sparse_depth_source) + ' (default: rgb)')

    parser.add_argument('--sparse-conf-type', metavar='TYPE', default='rgb',
                        choices=sparse_conf_source,
                        help='dataset: ' + ' | '.join(sparse_conf_source) + ' (default: rgb)')

    parser.add_argument('--max-gt-depth', default=math.inf, type=float, metavar='D',
                        help='cut-off depth of ground truth, negative values means infinity (default: inf [m])')

    # only valid for the fd input
    parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                        help='number of sparse depth samples (default: 0)')
    parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names,
                        help='sparsifier: ' + ' | '.join(sparsifier_names) + ' (default: ' + UniformSampling.name + ')')

    #input filter

    parser.add_argument('--min-depth', default=0.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--scaling', default='per_frame', choices=data_scale,
                        help='model architecture: ' + ' | '.join(data_scale) + ' (default: per_frame)')
    parser.add_argument('--global-divider', default=1.0, type=float, metavar='D',
                        help='Normalization factor (default: 1.0 [m])')





    #loss
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')

    #training params
    parser.add_argument('-o', '--optimizer', metavar='OPTIMIZER', default='sgd', choices=opt_names,
                        help='Optimizer: ' + ' | '.join(opt_names) + ' (default: SGD)')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
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

    #output
    parser.add_argument('--val-images', default=40, type=int, metavar='N',
                        help='number of images in the validation image (default: 40)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    #
    # args = parser.parse_args()


    # if args.pretrained == False:
    #     print("not using pretraining")
    # elif args.pretrained == 'resnet':
    #     args.pretrained = True
    # elif args.pretrained != '':
    #     assert os.path.isfile(args.pretrained)
    # else:
    #     args.pretrained = False
    #     print("not using pretraining")
    #
    # if not Modality.validate_static(args.modality):
    #     print("input modality with problem")
    #     exit(0)

    # if args.modality == 'rgb' and args.max_depth != 0.0:
    #     print("max depth is forced to be 0.0 when input modality is rgb/rgbd")
    #     args.max_depth = 0.0
    return parser

def create_data_loaders(data_path,data_type='visim',loader_type='val',arch='',sparsifier_type='uar',num_samples=500,modality='rgb-fd',depth_divider=1,max_depth=-1,max_gt_depth=-1,batch_size=8,workers=8):
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
        from dataloaders.kitti_dataloader import KITTIDataset

        dataset = KITTIDataset(data_path, type=loader_type,
            modality=modality, sparsifier=sparsifier, depth_divider=depth_divider, is_resnet= ('resnet' in arch),max_gt_depth=max_gt_depth)

    elif data_type == 'visim':
        from dataloaders.visim_dataloader import VISIMDataset

        dataset = VISIMDataset(data_path, type=loader_type,
            modality=modality, sparsifier=sparsifier, depth_divider=depth_divider, is_resnet= ('resnet' in arch),max_gt_depth=max_gt_depth)

    elif data_type == 'visim_seq':
        from dataloaders.visim_dataloader import VISIMSeqDataset
        dataset = VISIMSeqDataset(data_path, type=loader_type,
            modality=modality, sparsifier=sparsifier, depth_divider=depth_divider, is_resnet= ('resnet' in arch),max_gt_depth=max_gt_depth)
    else:
        raise RuntimeError('data type not found.' +
                           'The dataset must be either of kitti, visim or visim_seq.')

    if loader_type == 'val':
        # set batch size to be 1 for validation
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    elif loader_type == 'train':
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return loader, dataset

def main(args):

    output_directory = utils.get_output_directory(args)
    # evaluation mode
    start_epoch = 0
    if args.evaluate:
    #     assert os.path.isfile(args.evaluate), \
    #     "=> no best model found at '{}'".format(args.evaluate)
    #     print("=> loading best model '{}'".format(args.evaluate))
    #     checkpoint = torch.load(args.evaluate)
    #     output_directory = os.path.dirname(args.evaluate)
    #     old_args = args
    #     args = checkpoint['args']
    #     args.data_path = old_args.data_path
    #     start_epoch = checkpoint['epoch'] + 1
    #     best_result = checkpoint['best_result']
    #     model = checkpoint['model']
    #     print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
    #     _, val_loader = create_data_loaders(args)
    #     args.evaluate = True
    #     validate(val_loader, model,None, checkpoint['epoch'], write_to_file=False)
         return

    # optionally resume from a checkpoint
    elif args.resume:
        a = 1
        # if args.resume == 'continue' or args.resume == 'best':
        #     if args.resume == 'continue':
        #         pattern = 'checkpoint-*.pth.tar'
        #     if args.resume == 'best':
        #         pattern = 'model_best.pth.tar'
        #
        #     filename_regex = os.path.join(output_directory,pattern)
        #     possibilities = glob.glob(filename_regex)
        #     if len(possibilities) > 0 :
        #         possibilities.sort(reverse=True)
        #         args.resume = possibilities[0]
        #     else:
        #         raise RuntimeError("No checkpoint found at '{}'".format(output_directory))
        #
        #
        # assert os.path.isfile(args.resume), \
        #     "=> no checkpoint found at '{}'".format(args.resume)
        # print("=> loading checkpoint '{}'".format(args.resume))
        # checkpoint = torch.load(args.resume)
        # old_args = args
        # args = checkpoint['args']
        # args.data_path = old_args.data_path
        # start_epoch = checkpoint['epoch'] + 1
        # best_result = checkpoint['best_result']
        # model = checkpoint['model']
        # optimizer = checkpoint['optimizer']
        # output_directory = os.path.dirname(os.path.abspath(args.resume))
        # print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        # train_loader, val_loader = create_data_loaders(args)
        # args.resume = True

    # create new model
    else:


        criterion = create_loss(args)
        train_loader, val_loader = create_data_loaders(args)
        model, opt_parameters = create_model(args,val_loader)

        print("=> creating Model ({}-{}-{}) ...".format(args.arch, args.decoder, args.depth_weight_head_type))
        in_channels = g_modality.num_channels()
        opt_parameters = None

        print("=> model created. GPUS:{}".format(torch.cuda.device_count()))
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(opt_parameters, args.lr, \
                                        momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(opt_parameters, args.lr)
        else:
            raise RuntimeError ('unknow optimizer "{}"'.format(args.optimizer))

        if torch.cuda.device_count() > 1 :
            model = torch.nn.DataParallel(model) # for multi-gpu training
        model = model.cuda()
        summary(model,(4,240,320))

    # define loss function (criterion) and optimizer








    # create results folder, if not already exists

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')
    params_log = os.path.join(output_directory, 'params.txt')

    # create new csv files with only header
    if not args.resume:
        with open(params_log, 'w') as paramsfile:
            for arg, value in sorted(vars(args).items()):
                if isinstance(value,torch.nn.Module):
                    value = 'nn.Module argument'
                paramsfile.write("{}: {}\n".format(arg, value))


        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr,args.lrs,args.lrm)
        print('#### lr: {}'.format(optimizer.param_groups[0]['lr']))

        train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
        result, img_merge = validate(val_loader, model,criterion, epoch) # evaluate on validation set


        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer,
        }, is_best, epoch, output_directory)





def train(train_loader, model, criterion, optimizer, epoch):
    return

def validate(val_loader, model,criterion, epoch, write_to_file=True):
    return

if __name__ == '__main__':

    parser = create_command_parser()
    args = parser.parse_args(sys.argv[1:])
    print(args)
    #main()
