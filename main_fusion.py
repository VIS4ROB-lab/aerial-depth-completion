import os
import time
import csv
import numpy as np
import math
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torchsummary import summary
cudnn.benchmark = True
import matplotlib.pyplot as plt

from models import ResNet
from model_ext import DepthCompletionNet,DepthWeightCompletionNet,ValidDepthCompletionNet
from model_dual import SingleDepthCompletionNet,EarlyFusionNet,LateFusionNet,build_no_grad_mask
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils
from dataloaders.dataloader_ext import Modality
import inverse_warp as iw

args = utils.parse_command()
print(args)
g_modality = Modality(args.modality)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time','loss0','loss1','loss2']
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
                modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)
        val_dataset = NYUDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)

    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)

    elif args.data == 'visim':
        traindir = args.data_path
        valdir = args.data_path
        from dataloaders.visim_dataloader import VISIMDataset
        if not args.evaluate:
            train_dataset = VISIMDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)
        val_dataset = VISIMDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)

    elif args.data == 'visim_seq':
        traindir = args.data_path
        valdir = args.data_path
        from dataloaders.visim_dataloader import VISIMSeqDataset
        if not args.evaluate:
            train_dataset = VISIMSeqDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier,depth_divider=args.depth_divider, arch=args.arch)
        val_dataset = VISIMSeqDataset(valdir, type='val',
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
    output_directory = utils.get_output_directory(args)
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


        if not isinstance(args.pretrained, bool):
            print('loading pretraining {}'.format(args.pretrained))
            checkpoint_premodel = torch.load(args.pretrained)
            args.pretrained = checkpoint_premodel['model']
            if isinstance(args.pretrained, torch.nn.DataParallel):
                args.pretrained = args.pretrained.module

        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}-{}) ...".format(args.arch, args.decoder, args.depth_weight_head_type))
        if 'fsdepthcompnet18' in args.arch :
            if args.arch == 'efsdepthcompnet18':
                # assert args.data == 'visim_seq','wrong type of dataloader'
                model_single = SingleDepthCompletionNet(layers=18, modality_format=g_modality.format,
                                                 pretrained=args.pretrained)

                model = EarlyFusionNet(model_single)

        else:
            in_channels = g_modality.num_channels()
            if args.arch == 'resnet50':
                model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                    in_channels=in_channels, pretrained=args.pretrained)
            elif args.arch == 'resnet18':
                model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                    in_channels=in_channels, pretrained=args.pretrained)
            elif args.arch == 'resnet34':
                model = ResNet(layers=34, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                    in_channels=in_channels, pretrained=args.pretrained)
            elif args.arch == 'depthcompnet18':
                model = DepthCompletionNet(layers=18,modality_format=g_modality.format,
                                           pretrained=args.pretrained)
            elif args.arch == 'depthcompnet34':
                model = DepthCompletionNet(layers=34,
                                           modality_format=g_modality.format,
                                           pretrained=args.pretrained)
            elif args.arch == 'depthcompnet50':
                model = DepthCompletionNet(layers=50,
                                           modality_format=g_modality.format,
                                           pretrained=args.pretrained)
            elif args.arch == 'vdepthcompnet18':
                model = ValidDepthCompletionNet(layers=18, modality_format=g_modality.format,
                                           pretrained=args.pretrained)
            elif args.arch == 'vdepthcompnet34':
                model = ValidDepthCompletionNet(layers=34,
                                           modality_format=g_modality.format,
                                           pretrained=args.pretrained)
            elif args.arch == 'vdepthcompnet50':
                model = ValidDepthCompletionNet(layers=50,
                                           modality_format=g_modality.format,
                                           pretrained=args.pretrained)
            elif args.arch == 'weightcompnet18':
                model = DepthWeightCompletionNet(layers=18,
                                           modality_format=g_modality.format,
                                           pretrained=args.pretrained,dw_head_type=args.depth_weight_head_type)
            elif args.arch == 'weightcompnet34':
                model = DepthWeightCompletionNet(layers=34,
                                           modality_format=g_modality.format,
                                           pretrained=args.pretrained,dw_head_type=args.depth_weight_head_type)
            elif args.arch == 'weightcompnet50':
                model = DepthWeightCompletionNet(layers=50,
                                           modality_format=g_modality.format,
                                           pretrained=args.pretrained,dw_head_type=args.depth_weight_head_type)
            elif args.arch == 'sdepthcompnet18':
                model = SingleDepthCompletionNet(layers=18,modality_format=g_modality.format,
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
    elif args.criterion == 'l2gn':
        criterion = criteria.MaskedL2GradNormalLoss().cuda()
    elif args.criterion == 'l2nv':
        criterion = criteria.MaskedL2NormalValidLoss().cuda()
    elif args.criterion == 'l1smooth':
        criterion = criteria.MaskedL1LossSmoothess().cuda()
    elif args.criterion == 'wl1smooth':
        criterion = criteria.MaskedWL1LossSmoothess().cuda()





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

    gpu_time = data_time = 0
    model.train() # switch to train mode
    end = time.time()
    intrinsics = iw.Intrinsics(752, 480, 455, 455, 0, 0)

    for i, (input, target, scale,transformations) in enumerate(train_loader):

        #torch.cuda.synchronize()
        data_time = 0 #time.time() - end

        # compute pred
        end = time.time()

        if 'efsdepthcompnet' in args.arch :

            num_loops = len(input)
            previous_frame = None

            for t in range(num_loops):

                #prepare input
                curr_input = input[t]
                mask = build_no_grad_mask(curr_input[:,3:4,:,:])
                confidence = 0.7 * mask #fake confidence
                input_vec = torch.cat([curr_input,confidence,mask],dim=1)#r,g,b,d,c,m
                input_vec = input_vec.cuda()

                #prepare target
                target_depth = target[t].cuda()

                #scale to [0,10]
                for cb in range(input_vec.shape[0]):
                    input_vec[cb, 0, :, :] *= scale[t][cb].float().cuda()
                    target_depth_scaled = target_depth * scale[t][cb].float().cuda()

                #run model
                pred,pred_features = model(input_vec,previous_frame)

                if i > 0:
                    if args.criterion == 'wl1smooth':
                        loss = criterion(pred, target_depth_scaled, input_vec[:, 3:4, :, :])
                    else:
                        loss = criterion(pred, target_depth_scaled, epoch)

                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        print('ignoring image, no valid pixel')
                        continue

                    optimizer.zero_grad()
                    loss.backward()  # compute gradient and do SGD step
                    optimizer.step()

                    # scale back
                    for cb in range(input_vec.shape[0]):
                        pred[cb, 0, :, :] /= scale[t][cb].float().cuda()

                    # measure accuracy and record loss
                    result = Result()
                    result.evaluate(pred.data, target_depth.data)
                    average_meter.update(result, gpu_time, data_time, criterion.loss, input_vec.size(0))
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
                              'Lg10={result.lg10:.3f}({average.lg10:.3f}) '
                              'Loss={losses[0]}/{losses[1]}/{losses[2]} '.format(
                            epoch, i + 1, len(train_loader), data_time=data_time,
                            gpu_time=gpu_time, result=result, average=average_meter.average(), losses=criterion.loss))


                if t < len(input)-1:
                    t_wct = transformations[t].float()
                    t_wct1 = transformations[t+1].float()
                    t_ct1_ct = t_wct #create size
                    for bi in range(t_wct.shape[0]):
                        t_ct1_ct[bi,:,:] = torch.matmul(iw.inverse_pose(t_wct1[bi,:,:]) , t_wct[bi,:,:])
                    r_mat,t_vec = iw.decompose4(t_ct1_ct)
                    with torch.no_grad():
                        rbg_projected_back, map_t1_t  = iw.homography_from(input[t+1][:,0:3,:,:].cuda(),pred[:,:,:,:],r_mat.cuda(),t_vec.cuda(),intrinsics.scale(pred.shape[2],pred.shape[3]).cuda())
                        photometric_error = (rbg_projected_back - input_vec[:,0:3,:,:]).abs().norm(dim=1,keepdim=True)
                        features_vec= torch.cat([pred,photometric_error,pred_features],dim=1)

                        # def rgb2grayscale(rgb):
                        #     return rgb[0, :, :] * 0.2989 + rgb[1, :, :] * 0.587 + rgb[2, :, :] * 0.114
                        #
                        # img1 = rbg_projected_back[0, :, :, :].cpu().numpy()
                        # img2 = input_vec[0, 0:3, :, :].cpu().numpy()
                        #
                        # img1[0, :, :] = rgb2grayscale(img1)*5
                        # img1[1, :, :] = rgb2grayscale(img2)*5
                        # img1[2, :, :] = 0
                        # imgplot = plt.imshow(img1.transpose([1, 2, 0]))
                        # plt.show()

                        # transformed_feature_vec = torch.zeros_like(features_vec)

                        map_t1_t.floor_()

                        # for bi in  range(features_vec.shape[0]):
                        #     for row in range(features_vec.shape[2]):
                        #         for col in range(features_vec.shape[3]):
                        #             new_u = map_t1_t[bi, row, col, 0].long()
                        #             new_v = map_t1_t[bi, row, col, 1].long()
                        #             if new_u >= features_vec.shape[2] or new_v >= features_vec.shape[3]:
                        #                 a =1
                        #                 #print('estranho: {} / {}'.format(new_u,new_v))
                        #             elif new_u >= 0 and new_v >= 0:
                        #                 transformed_feature_vec[bi,:,new_u,new_v] =  features_vec[bi,:,row,col]
                        #
                        #
                        # #set features for the next frame
                        # previous_frame = transformed_feature_vec

        else:
            if 'vdepthcompnet' in args.arch:
                input, target = input.cuda(), target.cuda()
                target_depth = target[:, 0:1, :, :]
                pred, valids = model(input)
                loss = criterion(pred, valids, target_depth, epoch)
            elif args.arch == 'sdepthcompnet':
                target_depth = target[:, 0:1, :, :]
                mask = build_no_grad_mask(target_depth)
                confidence = 0.7 * mask
                # valid_mask = ((target_depth > 0).detach())
                # mask = torch.zeros_like(target_depth)
                # confidence = torch.zeros_like(target_depth)
                # mask[valid_mask] = 1
                # confidence[valid_mask] = 0.7
                input = torch.cat([input, confidence, mask], dim=1)  # r,g,b,d,c,m
                input, target = input.cuda(), target.cuda()
                target_depth = target[:, 0:1, :, :]

                pred, pred_features = model(input)
                if args.criterion == 'wl1smooth':
                    loss = criterion(pred, target_depth, input[:, 3:4, :, :])  # .unsqueeze(1)
                else:
                    loss = criterion(pred, target_depth, epoch)

            else:
                input, target = input.cuda(), target.cuda()
                target_depth = target[:, 0:1, :, :]
                pred = model(input)
                loss = criterion(pred, target_depth,epoch)

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print('ignoring image, no valid pixel')
                continue

            optimizer.zero_grad()
            loss.backward() # compute gradient and do SGD step
            optimizer.step()

            #torch.cuda.synchronize()
            gpu_time = 0 #time.time() - end


            for cb in range(pred.shape[0]):
                pred[cb,:,:,:] *= scale[cb].float().cuda()
                target_depth[cb,:,:,:] *= scale[cb].float().cuda()



            # measure accuracy and record loss
            result = Result()
            result.evaluate(pred.data, target_depth.data)
            average_meter.update(result, gpu_time, data_time,criterion.loss, input.size(0))
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
                      'Lg10={result.lg10:.3f}({average.lg10:.3f}) '
                      'Loss={losses[0]}/{losses[1]}/{losses[2]} '.format(
                      epoch, i+1, len(train_loader), data_time=data_time,
                      gpu_time=gpu_time, result=result, average=average_meter.average(),losses=criterion.loss))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time,'loss0': avg.loss0,'loss1': avg.loss1,'loss2': avg.loss2})



def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    normal_eval = criteria.MaskedL2GradNormalLoss().cuda().eval()
    end = time.time()
    num_of_images = 40
    sample_step = math.floor(len(val_loader) / float(num_of_images))
    for i, (input, target,scale) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        #torch.cuda.synchronize()
        data_time = 0 #time.time() - end

        valids = None

        # compute output
        end = time.time()
        with torch.no_grad():
            if 'vdepthcompnet' in args.arch:
                pred, valids = model(input)
            elif 'sdepthcompnet' in args.arch:
                target_depth = target[:, 0:1, :, :]
                valid_mask = (target_depth > 0)
                mask = torch.zeros_like(target_depth)
                confidence = torch.zeros_like(target_depth)
                mask[valid_mask] = 1
                confidence[valid_mask] = 0.7
                input = torch.cat([input, confidence, mask], dim=1)  # r,g,b,d,c,m
                input = input.cuda()
                pred,_ = model(input)
            else:
                pred = model(input)
            #torch.cuda.synchronize()
            gpu_time = 0 #time.time() - end

            target_depth = target[:, 0:1, :, :]
            #target_normal = target[:, 1:4, :, :]


            pred[0, :, :, :] *= scale[0]
            target_depth[0, :, :, :] *= scale[0]

            normal_eval(pred, target_depth)
            pred_normal, target_normal = normal_eval.get_extra_visualization()

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target_depth.data)
        average_meter.update(result, gpu_time, data_time,normal_eval.loss, input.size(0))
        end = time.time()

        skip = sample_step
        # save 8 images for visualization
        #skip = 1350
        # if args.modality == 'd':
        #     img_merge = None
        # else:
        #     if args.modality == 'rgb':
        #         rgb = input
        #     else:
        image_nchannels,_ = g_modality.get_input_image_channel()

        if image_nchannels == 3:
            rgb = input[:, 0:3, :, :]
        elif image_nchannels == 1:
            rgb = input[:, 0:1, :, :].data
            rgb = rgb.expand(-1, 3, -1, -1)
        else:
            rgb = torch.zeros(input[:, 0:1, :, :].size()).expand(-1,3,-1,-1)

        depth_nchannels,_ = g_modality.get_input_depth_channel()
        if(depth_nchannels == 1):
            depth = input[:,image_nchannels:(image_nchannels+1),:,:]*scale[0]
        else:
            depth = torch.zeros(input[:, 0:1, :, :].size())*scale[0]

        target_img = target_depth
        pred_img = pred

        if i == 0:
            img_merge = utils.merge_into_row_with_gt(rgb, depth, target_img, pred_img,target_normal,pred_normal,valids)
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)
        elif (i < num_of_images*skip) and (i % skip == 0):
            row = utils.merge_into_row_with_gt(rgb, depth, target_img, pred_img,target_normal,pred_normal,valids)
            img_merge = utils.add_row(img_merge, row)

            if (i % (4*skip) == 0):
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Loss={loss0}/{loss1}/{loss2} '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average(),loss0=normal_eval.loss[0],loss1=normal_eval.loss[1],loss2=normal_eval.loss[2]))

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
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time,'loss0': avg.loss0,'loss1': avg.loss1,'loss2': avg.loss2})
    return avg, img_merge

if __name__ == '__main__':
    main()
