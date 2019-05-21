
import main_conf
import dataloaders.dataloader_factory as df
import model_zoo.model_conf as mc



def test_main():
    args_list = ['-a asd','-l']
    parser = main_conf.create_command_parser()
    args = parser.parse_args(args_list)
    main_conf.main(args)


def test_raw():
    val_loader, _ = main_conf.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_small_v11', loader_type='val')
    train_loader, _ = main_conf.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_small_v11', loader_type='train')

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



if __name__ == '__main__':
    # service.py executed as script
    # do something
    print('hello')

    val_loader,_ = df.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_big_v8',loader_type='val')
    train_loader,_ = df.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_big_v8',loader_type='train')

    cdf =  mc.ConfidenceDepthFrameworkFactory()
    cdfmodel = cdf.create_model('rgbd','dc1-cf1-ln1','resnet18',None,'cbr3-c1',None,'resnet18',None)
    cdfmodel = cdfmodel.cuda()

    optimizer, scheduler = main_conf.create_optimizer('adam',cdfmodel.opt_params(),0,0,0.0001,5,0.1)

    loss = cdf.create_loss('l2',True,0.5).cuda()

    for epoch in range(1, 30):
        main_conf.train(train_loader,cdfmodel,loss,optimizer,'./res',epoch)
        main_conf.validate(val_loader, cdfmodel, loss, epoch, './res')
        scheduler.step(epoch)

    # for epoch in range(1, 10):
    #     for i, (input, target, scale) in enumerate(train_loader):
    #         input_cu = input.cuda()
    #         target_cu = target.cuda()
    #         d1, c1, d2 = cdfmodel(input_cu[:,:4,:,:])
    #         error = loss( input_cu[:,3:4,:,:], d1, d2, target_cu, 0)
    #         print(loss.loss)
    #         error.backward()
    #         print('.')
    #         optimizer.step()
    #     scheduler.step(epoch)
