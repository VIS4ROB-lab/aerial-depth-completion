import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torchsummary import summary
cudnn.benchmark = True

from models import ResNet
from model_ext import DepthCompletionNet
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils
from dataloaders.dataloader_ext import Modality

args = utils.parse_command()
print(args)
g_modality = Modality(args.modality)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join(args.data_path, args.data, 'train')
    valdir = os.path.join(args.data_path, args.data, 'val')
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)

    if args.data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'visim':
        traindir = args.data_path
        valdir = args.data_path
        from dataloaders.visim_dataloader import VISIMDataset
        if not args.evaluate:
            train_dataset = VISIMDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)
        val_dataset = VISIMDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_data_loaders(args)
        args.evaluate = True
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # optionally resume from a checkpoint
    elif args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(args.resume))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)
        args.resume = True

    # create new model
    else:

        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = g_modality.num_channels()
        if args.arch == 'resnet50':
            model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'depthcompnet18':
            model = DepthCompletionNet(layers=18,
                                       #output_size=train_loader.dataset.output_size,
                           #in_channels=in_channels,
                                       pretrained=args.pretrained)
        elif args.arch == 'depthcompnet34':
            model = DepthCompletionNet(layers=34,
                                       #output_size=train_loader.dataset.output_size,
                           #in_channels=in_channels,
                                       pretrained=args.pretrained)
        elif args.arch == 'depthcompnet50':
            model = DepthCompletionNet(layers=50,
                                       #output_size=train_loader.dataset.output_size,
                           #in_channels=in_channels,
                                       pretrained=args.pretrained)

        print("=> model created. GPUS:{}".format(torch.cuda.device_count()))
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)

        if torch.cuda.device_count() > 1 :
            model = torch.nn.DataParallel(model) # for multi-gpu training
        model = model.cuda()
        # summary(model,(4,240, 320))

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
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
                paramsfile.write("{}: {}\n".format(arg, value))


        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr,args.lrs)
        print('#### lr: {}'.format(optimizer.param_groups[0]['lr']))
        print(-1)
        train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
        result, img_merge = validate(val_loader, model, epoch) # evaluate on validation set


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
    average_meter = AverageMeter()
    print(-0.5)
    model.train() # switch to train mode
    end = time.time()
    print(0)
    for i, (input, target) in enumerate(train_loader):
        print(1)
        input, target = input.cuda(), target.cuda()
        print(2)
        #torch.cuda.synchronize()
        data_time = 0 #time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        print(3)
        loss = criterion(pred, target)
        print(4)
        if loss is None:
            print('ignoring image, no valid pixel')
            continue
        print(5)
        optimizer.zero_grad()
        print(6)
        loss.backward() # compute gradient and do SGD step
        print(7)
        optimizer.step()
        print(8)
        #torch.cuda.synchronize()
        gpu_time = 0 #time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data*args.depth_divider, target.data*args.depth_divider)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        #torch.cuda.synchronize()
        data_time = 0 #time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        #torch.cuda.synchronize()
        gpu_time = 0 #time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data*args.depth_divider, target.data*args.depth_divider)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 10
        # if args.modality == 'd':
        #     img_merge = None
        # else:
        #     if args.modality == 'rgb':
        #         rgb = input
        #     else:
        rgb = input[:,:3,:,:]
        depth = input[:,3:4,:,:]*args.depth_divider
        target_img = target*args.depth_divider
        pred_img = pred * args.depth_divider

        if i == 0:
            img_merge = utils.merge_into_row_with_gt(rgb, depth, target_img, pred_img)
        elif (i < 8*skip) and (i % skip == 0):
            row = utils.merge_into_row_with_gt(rgb, depth, target_img, pred_img)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8*skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()
