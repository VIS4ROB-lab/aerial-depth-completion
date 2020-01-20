import sys
import trainer
import dataloaders.dataloader_factory as df
import model_zoo.confidence_depth_framework as mc
import torch
import os
import math
import time



def create_output_folder(args):
    if isinstance(args.dcnet_pretrained, bool):
        pretrain_text= str(args.dcnet_pretrained)
    else:
        parts = args.dcnet_pretrained.split('@')
        pretrain_text = parts[-1]
        if(len(pretrain_text) > 10):
            pretrain_text = pretrain_text[-10:]

    current_time = time.strftime('%Y-%m-%d@%H-%M-%S')
    if args.output:
        output_directory = os.path.join('results', '{}_{}'.format(args.output,current_time))
    else:
        output_directory = os.path.join('results',
            '{}.spl={}.mod={}.inp={}.overall={}.dcnet={}.confnet={}.lossnet={}.crit={}.div={}.lr={}.lrs={}.bs={}.pre={}.time={}'.
            format(args.data_type, args.num_samples,args.data_modality,args.dcnet_modality, args.training_mode, \
                args.dcnet_arch,args.confnet_arch,args.lossnet_arch,  args.criterion, args.divider, args.lr,args.lrs, args.batch_size, \
                pretrain_text,current_time))
    return output_directory


def create_eval_output_folder(args):
    current_time = time.strftime('%Y-%m-%d@%H-%M-%S')
    if args.output:
        output_directory = os.path.join('results', '{}_{}'.format(args.output,current_time))
    else:
        output_directory = os.path.join('results','eval.time={}'.format(current_time))
    return output_directory


def save_arguments(args,output_folder):
    with open(os.path.join(output_folder,'params.txt'), 'w') as paramsfile:
        for arg, value in sorted(vars(args).items()):
            if isinstance(value, torch.nn.Module):
                value = 'nn.Module argument'
            paramsfile.write("{}: {}\n".format(arg, value))


def main_func(args):

    cdf = mc.ConfidenceDepthFrameworkFactory()
    val_loader, _ = df.create_data_loaders(args.data_path
                                           , loader_type='val'
                                           , data_type= args.data_type
                                           , modality= args.data_modality
                                           , num_samples= args.num_samples
                                           , depth_divisor= args.divider
                                           , max_depth= args.max_depth
                                           , max_gt_depth= args.max_gt_depth
                                           , workers= args.workers
                                           , batch_size=1)
    if not args.evaluate:
        train_loader, _ = df.create_data_loaders(args.data_path
                                                 , loader_type='train'
                                                 , data_type=args.data_type
                                                 , modality=args.data_modality
                                                 , num_samples=args.num_samples
                                                 , depth_divisor=args.divider
                                                 , max_depth=args.max_depth
                                                 , max_gt_depth=args.max_gt_depth
                                                 , workers=args.workers
                                                 , batch_size=args.batch_size)

    # evaluation mode
    if args.evaluate:
        cdfmodel,loss, epoch = trainer.resume(args.evaluate,cdf,True)
        output_directory = create_eval_output_folder(args)
        os.makedirs(output_directory)
        print(output_directory)
        save_arguments(args,output_directory)
        trainer.validate(val_loader, cdfmodel, loss, epoch,print_frequency=args.print_freq,num_image_samples=args.val_images, output_folder=output_directory, conf_recall=args.pr,conf_threshold= args.thrs)
        return

    output_directory = create_output_folder(args)
    os.makedirs(output_directory)
    print(output_directory)
    save_arguments(args, output_directory)

    # optionally resume from a checkpoint
    if args.resume:
        cdfmodel, loss, loss_def, best_result_error, optimizer, scheduler = trainer.resume(args.resume,cdf,False)

    # create new model
    else:
        cdfmodel = cdf.create_model(args.dcnet_modality, args.training_mode, args.dcnet_arch, args.dcnet_pretrained, args.confnet_arch, args.confnet_pretrained, args.lossnet_arch, args.lossnet_pretrained)
        cdfmodel, opt_parameters = cdf.to_device(cdfmodel)
        optimizer, scheduler = trainer.create_optimizer(args.optimizer, opt_parameters, args.momentum, args.weight_decay, args.lr, args.lrs, args.lrm)
        loss, loss_definition = cdf.create_loss(args.criterion, ('ln' in args.training_mode), (0.5 if 'dc1' in args.training_mode else 1.0))
        best_result_error = math.inf


    for epoch in range(0, args.epochs):
        trainer.train(train_loader, cdfmodel, loss, optimizer, output_directory, epoch)
        epoch_result = trainer.validate(val_loader, cdfmodel, loss, epoch=epoch,print_frequency=args.print_freq,num_image_samples=args.val_images, output_folder=output_directory)
        scheduler.step(epoch)

        is_best = epoch_result.rmse < best_result_error
        if is_best:
            best_result_error = epoch_result.rmse
            trainer.report_top_result(os.path.join(output_directory, 'best_result.txt'), epoch, epoch_result)
            # if img_merge is not None:
            #     img_filename = output_directory + '/comparison_best.png'
            #     utils.save_image(img_merge, img_filename)

        trainer.save_checkpoint(cdf, cdfmodel, loss_definition, optimizer, scheduler,best_result_error, is_best, epoch,
                                output_directory)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        trainer.create_command_parser().print_help()
        exit(0)

    arg_list = sys.argv[1:]

    arg_parser = trainer.create_command_parser()
    args = arg_parser.parse_args(arg_list)
    print(args)
    main_func(args)


