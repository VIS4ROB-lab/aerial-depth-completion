
import trainer
import dataloaders.dataloader_factory as df
import model_zoo.confidence_depth_framework as mc
import torch
import math



def test_main():
    args_list = ['-a asd','-l']
    parser = trainer.create_command_parser()
    args = parser.parse_args(args_list)
    trainer.main(args)


def test_raw():
    val_loader, _ = trainer.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_small_v11', loader_type='val')
    train_loader, _ = trainer.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_small_v11', loader_type='train')

    cdf = mc.ConfidenceDepthFrameworkFactory()
    cdfmodel = mc.ConfidenceDepthFrameworkModel()

    dc_model = cdf.create_dc_model(model_arch='resnet18', pretrained_args='resnet', input_type='rgbd',
                                   output_type='dc').cuda()
    conf_model = cdf.create_conf_model(model_arch='forward', pretrained_args=None, dc_model=dc_model).cuda()
    loss_dc_model = cdf.create_dc_model(model_arch='resnet18', pretrained_args='resnet', input_type='rgbdc',
                                        output_type='d').cuda()

    cdfmodel.dc_model = dc_model
    cdfmodel.conf_model = conf_model
    cdfmodel.loss_dc_model = loss_dc_model

    loss = cdf.create_loss('l2', True, 0.5).cuda()

    for i, (input, target, scale) in enumerate(train_loader):
        input_cu = input.cuda()
        target_cu = target.cuda()
        d1, c1, d2 = cdfmodel(input_cu[:, :4, :, :])
        error = loss(input_cu[:, 3:4, :, :], d1, d2, target_cu, 0)
        print(loss.loss)
        error.backward()
        print('.')

# def save_checkpoint(cdfmodel, optimizer, filename):
#     # optionally resume from a checkpoint
#     if resume:
#         if os.path.isfile(resume):
#             print_log("=> loading checkpoint '{}'".format(resume), log)
#             checkpoint = torch.load(resume)
#             recorder = checkpoint['recorder']
#             start_epoch = checkpoint['epoch']
#             scheduler.load_state_dict(checkpoint['scheduler'])
#             net.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             print_log("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']), log)
#         else:
#             print_log("=> no checkpoint found at '{}'".format(resume), log)
#     else:
#         print_log("=> did not use any checkpoint for {} model".format(arch), log)

# def save_checkpoint(cdfmodel, optimizer,scheduler, filename):



if __name__ == '__main__':
    # service.py executed as script
    # do something
    print('hello')

    val_loader,_ = df.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_small_v11',loader_type='val',batch_size=1)
    train_loader,_ = df.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_small_v11',loader_type='train',batch_size=2)

    cdf =  mc.ConfidenceDepthFrameworkFactory()
    cdfmodel = cdf.create_model('rgbd','dc1-cf0-ln1','resnet18',None,'cbr3-c1',None,'resnet18',None)
    cdfmodel,opt_parameters = cdf.to_device(cdfmodel)

    optimizer, scheduler = trainer.create_optimizer('adam', opt_parameters, 0, 0, 0.00001, 100, 0.1)

    loss,loss_definition = cdf.create_loss('l2',True,0.5)
    # main_conf.save_checkpoint(cdf,cdfmodel,loss_definition,optimizer,scheduler,True,epoch,output_directory)
    # loss = loss.cuda()
    # model_state = cdf.get_state(cdfmodel)
    # optimizer_state = main_conf.get_optimizer_state(optimizer, scheduler)
    # checkpoint={}
    # checkpoint['model_state'] = model_state
    # checkpoint['optimizer_state'] = optimizer_state
    # checkpoint['loss_definition'] = loss_definition
    # torch.save(checkpoint, '/media/lucas/lucas-ds2-1tb/log_big_data1.pth.tar')
    # checkload = torch.load('/media/lucas/lucas-ds2-1tb/log_big_data1.pth.tar')
    #
    # loss2,loss_def2 = cdf.create_loss_fromstate(checkload['loss_definition'])
    # loss2 = loss2.cuda()
    # cdfmodel2 = cdf.create_model_from_state(checkload['model_state']).cuda()
    # optimizer2,scheduler2 = main_conf.create_optimizer_fromstate(cdfmodel2.opt_params(),checkload['optimizer_state'])
    output_directory = './res1'
    best_result = math.inf
    for epoch in range(1, 30):
        trainer.train(train_loader, cdfmodel, loss, optimizer, output_directory, epoch)
        epoch_result = trainer.validate(val_loader, cdfmodel, loss, epoch, output_directory)
        scheduler.step(epoch)

        is_best = epoch_result.rmse < best_result
        if is_best:
            best_result = epoch_result.rmse
            trainer.report_top_result(output_directory + '/best_result.txt', epoch, epoch_result)
            # if img_merge is not None:
            #     img_filename = output_directory + '/comparison_best.png'
            #     utils.save_image(img_merge, img_filename)

        trainer.save_checkpoint(cdf, cdfmodel, loss_definition, optimizer, scheduler, is_best, epoch, output_directory)



