
import main_conf
from model_zoo.model_conf import ConfidenceDepthFrameworkModel



def test_main():
    args_list = ['-a asd','-l']
    parser = main_conf.create_command_parser()
    args = parser.parse_args(args_list)
    main_conf.main(args)


if __name__ == '__main__':
    # service.py executed as script
    # do something
    print('hello')

    val_loader,_ = main_conf.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_small_v11',loader_type='val')
    train_loader,_ = main_conf.create_data_loaders('/media/lucas/lucas-ds2-1tb/dataset_small_v11',loader_type='train')

    cdf =  main_conf.ConfidenceDepthFrameworkFactory()
    cdfmodel = ConfidenceDepthFrameworkModel()

    dc_model = cdf.create_dc_model(model_arch='resnet18', pretrained_args='resnet', input_type='rgbd', output_type='dc').cuda()
    conf_model = cdf.create_conf_model(model_arch='forward', pretrained_args=None, dc_model=dc_model).cuda()
    loss_dc_model = cdf.create_dc_model(model_arch='resnet18', pretrained_args='resnet', input_type='rgbdc', output_type='d').cuda()

    cdfmodel.dc_model = dc_model
    # cdfmodel.conf_model = conf_model
    # cdfmodel.loss_dc_model = loss_dc_model

    loss = cdf.create_loss('l2',False,0.5).cuda()

    for i, (input, target, scale) in enumerate(train_loader):
        input_cu = input.cuda()
        target_cu = target.cuda()
        d1,c1 = dc_model(input_cu[:,:4,:,:],False)
        error = loss(input_cu[:,3:4,:,:],d1,target_cu,0)
        error.backward()
        print('.')
