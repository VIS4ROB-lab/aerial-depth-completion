import os
import sys
import csv
import time

import math
import utils
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import GPUtilext
import torch.optim
from torchsummary import summary
from metrics import AverageMeter, Result,ConfidencePixelwiseAverageMeter,ConfidencePixelwiseThrAverageMeter


cudnn.benchmark = True



def create_command_parser():

    import argparse

    model_names = ['resnet18', 'udepthcompnet18','erfdepthcompnet','gms_depthcompnet','ged_depthcompnet']
    model_input_type = ['rgb','rgbd','rgbdw'] #['d','dw','c','cd','cdw']
    #image_type_source = ['g', 'rgb','undefined']
    #sparse_depth_source = ['kgt','kor','kde','fd','undefined']
    #sparse_conf_source = ['bin', 'kw','undefined']
    training_mode = ['dc1_only','dc1-ln0','dc1-ln1', 'dc0-cf1-ln0', 'dc1-cf1-ln0', 'dc0-cf1-ln1', 'dc1-cf1-ln1']
    confnet_exclusive_names = ['cbr3-c1','cbr3-cbr1-c1', 'cbr3-cbr1-c1res' ]#'cbr3-cbr1-c1','cbr5-cbr3-cbr1-c1'
    confnet_names = confnet_exclusive_names + ['join','none']
    data_modality_types = ['rgb-fd-bin','rgb-kfd-bin','rgb-kgt-bin','rgb-kor-bin','rgb-kor-kw']
    #confnet_input_type = ['c','cd','cw','cdw','cdwr', 'cdwrl','cdr', 'cdrl', 'clr','lr'] #'c','cd','cw','cdw','cdwr', 'cdwrl','cdr', 'cdrl', 'clr'

    loss_names = ['l1', 'l2', 'il1', 'absrel']
    data_types = ['nyuv2', 'visim', 'visim_nyuv2', 'kitti' ]
    #data_scale = ['per_frame', 'global', 'none']

    opt_names = ['sgd', 'adam']
    from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
    sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo]]

    parser = argparse.ArgumentParser(description='Confidence Depth Completion')

    # training
    parser.add_argument('--output', metavar='FOLDER', default='',
                        help='output folder')
    parser.add_argument('--training-mode', metavar='ARCH', default='dc1', choices=training_mode,
                        help='training_mode: ' + ' | '.join(training_mode) + ' (default: dc1)')

    #dcnet
    parser.add_argument('--dcnet-arch', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')

    parser.add_argument('--dcnet-pretrained', default='', type=str, metavar='PATH',
                        help='path to pretraining checkpoint (default: )')

    parser.add_argument('--dcnet-modality', metavar='MODALITY', default='rgbd', choices=model_input_type, type=str,
                        help='modality: ' + ' | '.join(model_input_type) + ' (default: rgbd)')

    #confnet
    parser.add_argument('--confnet-arch', metavar='ARCH', default='cbr3-c1', choices=confnet_names,
                        help='model architecture: ' + ' | '.join(confnet_names) + ' (default: cbr3-c1)')

    parser.add_argument('--confnet-pretrained', default='', type=str, metavar='PATH',
                        help='path to pretraining checkpoint (default: )')
    #lossnet
    parser.add_argument('--lossnet-arch', metavar='ARCH', default='ged_depthcompnet', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: ged_depthcompnet)')

    parser.add_argument('--lossnet-pretrained', default='', type=str, metavar='PATH',
                        help='path to pretraining checkpoint (default: )')

    #input data
    parser.add_argument('--data-type', metavar='DATA', default='visim',
                        choices=data_types, help='dataset: ' + ' | '.join(data_types) + ' (default: visim)')
    parser.add_argument('--data-path', default='data', type=str, metavar='PATH',
                        help='path to data folder')
    parser.add_argument('--data-modality', metavar='MODALITY', default='rgb-fd-bin', choices=data_modality_types,
                        type=str, help='modality: ' + ' | '.join(data_modality_types) + ' (default: rgb-fd-bin)')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--max-gt-depth', default=math.inf, type=float, metavar='D',
                        help='cut-off depth of ground truth, negative values means infinity (default: inf [m])')
    parser.add_argument('--min-depth', default=0.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier (default: 0 [m])')
    parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--divider', default=0, type=float, metavar='D',
                        help='Normalization factor - zero means per frame (default: 0 [m])')

    # only valid for the fd input
    parser.add_argument('-s', '--num-samples', default=500, type=int, metavar='N',
                        help='number of sparse depth samples (default: 500)')
    parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names,
                        help='sparsifier: ' + ' | '.join(sparsifier_names) + ' (default: ' + UniformSampling.name + ')')

    #loss
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')

    #training params
    parser.add_argument('-o', '--optimizer', metavar='OPTIMIZER', default='adam', choices=opt_names,
                        help='Optimizer: ' + ' | '.join(opt_names) + ' (default: adam)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='mini-batch size (default: 8)')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,dest='lr',
                        metavar='LR', help='initial learning rate (default 0.001)')
    parser.add_argument('-lrs', '--learning-rate-step', default=5, type=int, metavar='LRS',dest='lrs',
                        help='number of epochs between reduce the learning rate by 10 (default: 5)')
    parser.add_argument('-lrm', '--learning-rate-multiplicator', default=0.1, type=float, dest='lrm',
                        metavar='LRM', help='multiplicator (default 0.1)')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')

    #output
    parser.add_argument('--val-images', default=10, type=int, metavar='N',
                        help='number of images in the validation image (default: 40)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    #alternative modes
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help="path to latest checkpoint (default: empty)")
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='', metavar='PATH',
                        help='evaluate model on validation set (default: empty)')
    parser.add_argument('-pr', '--precision-recall', dest='pr', default=False, action='store_true',
                        help='calculate the pr')

    return parser


def create_optimizer(optimizer_type, parameters, momentum=0, weight_decay=0, lr_init=10e-4, lr_step=5, lr_gamma=0.1):

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(params=parameters, lr=lr_init, \
                                    momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params=parameters, lr=lr_init,weight_decay=weight_decay)
    else:
        raise RuntimeError('unknow optimizer "{}"'.format(optimizer_type))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    return optimizer, scheduler

def get_optimizer_state(optimizer, scheduler):
    state = {}
    state['optimizer_type'] = ('adam' if isinstance(optimizer,torch.optim.Adam) else 'sgd')
    state['optimizer_state'] = optimizer.state_dict()
    state['scheduler_state'] = scheduler.state_dict()
    return state

def create_optimizer_fromstate(parameters,state):
    optimizer, scheduler = create_optimizer(state['optimizer_type'],parameters)
    optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state'])
    return optimizer, scheduler

def resume(filename, factory,only_evaluation):
    checkpoint = torch.load(filename)
    loss, loss_def = factory.create_loss_fromstate(checkpoint['loss_definition'])
    cdfmodel = factory.create_model_from_state(checkpoint['model_state'])
    cdfmodel, opt_parameters = factory.to_device(cdfmodel)
    epoch = checkpoint['epoch']
    if not only_evaluation:
        best_result_error = checkpoint['best_result_error']
        optimizer, scheduler = create_optimizer_fromstate(opt_parameters, checkpoint['optimizer_state'])
        return cdfmodel, loss, loss_def, best_result_error, optimizer, scheduler

    return cdfmodel,loss,epoch


def save_checkpoint(factory,cdfmodel,loss_definition,optimizer, scheduler,best_result_error,is_best,epoch,output_directory):

    model_state = factory.get_state(cdfmodel)
    optimizer_state = get_optimizer_state(optimizer, scheduler)
    checkpoint = {  'model_state': model_state,
                    'optimizer_state': optimizer_state,
                    'loss_definition':loss_definition,
                    'epoch':epoch,
                    'best_result_error':best_result_error}
    utils.save_checkpoint(checkpoint,is_best,epoch,output_directory)


def train(train_loader, model, criterion, optimizer,output_folder,  epoch):

    average_meter = [AverageMeter(),AverageMeter()]

    model.train()  # switch to train mode
    end = time.time()
    num_total_samples = len(train_loader)
    for i, (input, target, scale) in enumerate(train_loader):

        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()

        input, target = input.cuda(), target.cuda()
        target_depth = target[:, 0:1, :, :]
        prediction = model(input)
        if prediction[2] is not None: #d1,c1,d2
            loss = criterion(input, prediction[0][:, 0:1, :, :], prediction[2][:, 0:1, :, :], target_depth, epoch)
        else:
            loss = criterion(input, prediction[0][:, 0:1, :, :], target_depth, epoch)

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            print('ignoring image, no valid pixel')
            continue

        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        for cb in range(prediction[0].size(0)):
            prediction[0][cb, :, :, :] *= scale[cb]
            if prediction[2] is not None:
                prediction[2][cb, :, :, :] *= scale[cb]
            target_depth[cb, :, :, :] *= scale[cb]

        # measure accuracy and record loss
        result = [Result(),Result()]
        result[0].evaluate(prediction[0][:, 0:1, :, :].data, target_depth.data)
        average_meter[0].update(result[0], gpu_time, data_time, criterion.loss, input.size(0))
        if prediction[2] is not None:
            result[1].evaluate(prediction[2][:, 0:1, :, :].data, target_depth.data)
            average_meter[1].update(result[1], gpu_time, data_time, criterion.loss, input.size(0))


        end = time.time()

        if (i + 1) % 10 == 0:
            print_error('Train',num_total_samples, average_meter[0].average(), result[0], criterion.loss, data_time, gpu_time, i, epoch)
            if prediction[2] is not None:
                print_error('Train',num_total_samples, average_meter[1].average(), result[1], criterion.loss, data_time, gpu_time, i, epoch)

    report_epoch_error(os.path.join(output_folder,'train.csv'), epoch, average_meter[0].average())
    if prediction[2] is not None:
        report_epoch_error(os.path.join(output_folder,'train.csv'), epoch, average_meter[1].average())


def report_top_result(filename_csv,epoch,epoch_result):
    with open(filename_csv, 'w') as txtfile:
        txtfile.write(
            "epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                format(epoch, epoch_result.mse, epoch_result.rmse, epoch_result.absrel, epoch_result.lg10,
                       epoch_result.mae, epoch_result.delta1,
                       epoch_result.gpu_time))


def report_epoch_error(filename_csv, epoch, avg):
    fieldnames = ['epoch', 'mse', 'rmse', 'absrel', 'lg10', 'mae',
                  'delta1', 'delta2', 'delta3',
                  'data_time', 'gpu_time', 'loss0', 'loss1', 'loss2']
    with open(filename_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'epoch': epoch, 'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                         'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                         'gpu_time': avg.gpu_time, 'data_time': avg.data_time, 'loss0': avg.loss0, 'loss1': avg.loss1,
                         'loss2': avg.loss2})


def print_error(type,num_total_samples, average, result, loss, data_time, gpu_time, i, epoch):
    # print('=> output: {}'.format(output_directory))
    print('{type} Epoch: {0} [{1}/{2}]\t'
          't_Data={data_time:.3f}({average.data_time:.3f}) '
          't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
          'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
          'MAE={result.mae:.2f}({average.mae:.2f}) '
          'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
          'REL={result.absrel:.3f}({average.absrel:.3f}) '
          'Lg10={result.lg10:.3f}({average.lg10:.3f}) '
          'Loss={losses[0]}/{losses[1]}/{losses[2]} '.format(
        epoch, i + 1, num_total_samples, data_time=data_time,
        gpu_time=gpu_time, result=result, average=average, type=type, losses=loss))
    attrlist = [[
        {'attr': 'id', 'name': 'ID'},
        {'attr': 'load', 'name': 'GPU util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0},
        {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '%', 'transform': lambda x: x * 100,
         'precision': 0}],
        [{'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
         {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0},
         {'attr': 'memoryFree', 'name': 'Memory free', 'suffix': 'MB', 'precision': 0}]]
    GPUtilext.showUtilization(attrList=attrlist)


class ResultSampleImage():
    def __init__(self,output_directory,epoch, num_samples, total_images):
        self.normal_net = None
        self.image = None
        self.num_samples = num_samples
        self.sample_step = total_images // float(num_samples)
        self.filename = output_directory + '/comparison_' + str(epoch) + '.png'

    def save(self,input,prediction,target,to_disk=False):
        rgb = input[0,:3,:,:]
        input_depth = input[0,3:4,:,:]
        input_conf =  input[0,4:5,:,:]
        in_gt_depth = target[0,:1,:,:]
        out_depth1 = prediction[0][0,:1,:,:]
        if prediction[1] is not None:
            out_conf1 = prediction[1][0,:1,:,:]
        else:
            out_conf1 = None

        if prediction[2] is not None:
            out_depth2 = prediction[2][0,:1,:,:]
        else:
            out_depth2 = None


        row = utils.merge_into_row_with_gt2(rgb, input_depth,input_conf, in_gt_depth , out_depth1, out_conf1, out_depth2)
        if(self.image is not None):
            self.image = utils.add_row(self.image, row)
        else:
            self.image = row

        if to_disk:
            utils.save_image(self.image, self.filename)

    def update(self,i, input, prediction,target):
        if (i % self.sample_step) == 0:
            self.save(input, prediction, target,((i % 2*self.sample_step) == 0))



def validate(val_loader, model,criterion, epoch, num_image_samples=4, print_frequency=10, output_folder=None, conf_recall=False):
    average_meter = [AverageMeter(), AverageMeter()]

    if conf_recall:
        conf_avg_meter = ConfidencePixelwiseThrAverageMeter()

    model.eval()  # switch to train mode
    end = time.time()
    num_total_samples = len(val_loader)
    rsi = ResultSampleImage(output_folder,epoch,num_image_samples,num_total_samples)
    for i, (input, target, scale) in enumerate(val_loader):

        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()

        input, target = input.cuda(), target.cuda()
        target_depth = target[:, 0:1, :, :]
        prediction = model(input)
        if prediction[2] is not None:  # d1,c1,d2
            loss = criterion(input, prediction[0][:, 0:1, :, :], prediction[2][:, 0:1, :, :], target_depth, epoch)
        else:
            loss = criterion(input, prediction[0][:, 0:1, :, :], target_depth, epoch)

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            print('ignoring image, no valid pixel')
            continue

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        for cb in range(prediction[0].size(0)):
            prediction[0][cb, :, :, :] *= scale[cb]
            if prediction[2] is not None:
                prediction[2][cb, :, :, :] *= scale[cb]
            target_depth[cb, :, :, :] *= scale[cb]
            input[cb, 3:4, :, :] *= scale[cb]

        # measure accuracy and record loss
        result = [Result(), Result()]
        result[0].evaluate(prediction[0][:, 0:1, :, :].data, target_depth.data)
        average_meter[0].update(result[0], gpu_time, data_time, criterion.loss, input.size(0))
        if prediction[2] is not None:
            result[1].evaluate(prediction[2][:, 0:1, :, :].data, target_depth.data)
            average_meter[1].update(result[1], gpu_time, data_time, criterion.loss, input.size(0))

        end = time.time()

        if (i + 1) % print_frequency == 0:
            print_error('Val', num_total_samples, average_meter[0].average(), result[0],
                        criterion.loss, data_time, gpu_time, i, epoch)
            if prediction[2] is not None:
                print_error('Val', num_total_samples, average_meter[1].average(), result[1],
                            criterion.loss, data_time, gpu_time, i, epoch)

        if conf_recall and (i % 10 == 0):
            conf_avg_meter.evaluate(prediction[0][:, 0:1, :, :].data, prediction[1][:, 0:1, :, :].data, target_depth.data)

        rsi.update(i, input, prediction, target_depth)

    final_result = average_meter[0].average()
    report_epoch_error(os.path.join(output_folder, 'val.csv'), epoch, final_result)
    if prediction[2] is not None:
        report_epoch_error(os.path.join(output_folder, 'val.csv'), epoch, average_meter[1].average())

    if conf_recall:
        conf_avg_meter.print(os.path.join(output_folder, 'pr.csv'))

    return final_result


