import torch
import torch.nn as nn
import torch.nn.functional as F
from nconv_sd import CNN as unguided_net
from model_zoo.s2d_resnet import S2DResNet
import guided_ms_net


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
            model = CBR3C1Confidence(in_channels=in_channels)
        elif model_arch == 'forward':
            model = ForwardConfidence(in_channels=in_channels)
        else:
            raise RuntimeError('Dataset not found.' +
                               'The dataset must be either of nyudepthv2 or kitti.')

        return model

    def create_loss(self, criterion, dual=False, weight1=0):

        if criterion == 'l2':
            loss = MaskedMSELoss()
        elif criterion == 'l1':
            loss = MaskedL1Loss()
        elif criterion == 'il1':
            loss = InvertedMaskedL1Loss()
        elif criterion == 'absrel':
            loss = MaskedAbsRelLoss()
        if dual:
            loss = DualLoss(loss, loss, weight1)
        return loss

    def create_model(self, input_type, overall_arch, dc_arch, dc_weights, conf_arch=None
                     , conf_weights=None, lossdc_arch=None, lossdc_weights=None):


        cdfmodel = ConfidenceDepthFrameworkModel()

        cdfmodel.dc_model = None
        cdfmodel.conf_model = None
        cdfmodel.loss_dc_model = None
        cdfmodel.overall_arch = overall_arch

        if 'dc' in overall_arch:

            if 'only' in overall_arch or 'cf' in overall_arch:
                output_type = 'd'
            else:
                output_type = 'dc'

            cdfmodel.dc_model = self.create_dc_model(dc_arch, dc_weights, input_type , output_type)

        if 'cf' in overall_arch:
            cdfmodel.conf_model = self.create_conf_model(model_arch=conf_arch, pretrained_args=conf_weights, dc_model=cdfmodel.dc_model)

        if 'ln' in overall_arch:
            cdfmodel.loss_dc_model = self.create_dc_model(model_arch=lossdc_arch, pretrained_args=lossdc_weights, input_type='rgbdc',
                                                output_type='d')

        cdfmodel.input_size = len(input_type)



        return cdfmodel


def init_weights(m):
    #from Ma and Karaman 2018
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ConfidenceDepthFrameworkModel(torch.nn.Module):

    def __init__(self):
        super(ConfidenceDepthFrameworkModel, self).__init__()

        self.dc_model = None
        self.conf_model = None
        self.loss_dc_model = None
        self.overall_arch = ''
        self.input_size = 0 #acceptable inputs are 3:rgb, 4:rgbd, 5:rgbdc

    def forward(self, input): #input rgbdc
        dc_x = input[:,:self.input_size,:,:]
        depth1,conf_x = self.dc_model(dc_x,(self.conf_model is not None))

        if self.conf_model is not None:
            assert(conf_x is not None),'dc_model does not support extern confidence net'
            conf1 = self.conf_model(conf_x)
        else:
            conf1 = None

        if self.loss_dc_model is not None:
            rgbd1c1= torch.cat([input[:,:3,:,:],depth1,conf1],dim=1)
            depth2,_ = self.loss_dc_model(rgbd1c1,False)
        else:
            depth2 = None

        return depth1, conf1, depth2

    def opt_params(self):
        opt_parameters = []

        if 'dc1' in self.overall_arch:
            assert self.dc_model is not None
            opt_parameters += self.dc_model.parameters()

        if 'cf1' in self.overall_arch:
            assert self.conf_model is not None
            opt_parameters += self.conf_model.parameters()

        if 'ln1' in self.overall_arch:
            assert self.loss_dc_model is not None
            opt_parameters += self.loss_dc_model.parameters()
        return opt_parameters


###########################################
#Confidence nets
###########################################

class CBR3C1Confidence(nn.Module):

    def __init__(self,in_channels):
        super(CBR3C1Confidence, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1,
                                1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.relu1 =nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels, 1, 1, 1,
                                0, bias=True)

        # initialize the weights
        init_weights(self.conv1)
        init_weights(self.bn1)
        init_weights(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return torch.sigmoid(x)


class ForwardConfidence(nn.Module):

    def __init__(self,in_channels):
        super(ForwardConfidence, self).__init__()

    def forward(self, x):
        assert(x.size()[1] == 1), 'forward confidence need to have only one channel'
        return x

################################################
# depth_completion nets
################################################


class GEDNet(nn.Module):

    def __init__(self, pos_fn='SoftPlus', pretrained=None):
        super(GEDNet,self).__init__()

        # Import the unguided network
        self.d_net = unguided_net(pos_fn)

        # U-Net
        self.conv1 = nn.Conv2d(5, 80, (3, 3), 2, 1, bias=True)
        self.conv2 = nn.Conv2d(80, 80, (3, 3), 2, 1, bias=True)
        self.conv3 = nn.Conv2d(80, 80, (3, 3), 2, 1, bias=True)
        self.conv4 = nn.Conv2d(80, 80, (3, 3), 2, 1, bias=True)
        self.conv5 = nn.Conv2d(80, 80, (3, 3), 2, 1, bias=True)

        self.conv6 = nn.Conv2d(80 + 80, 64, (3, 3), 1, 1, bias=True)
        self.conv7 = nn.Conv2d(64 + 80, 64, (3, 3), 1, 1, bias=True)
        self.conv8 = nn.Conv2d(64 + 80, 32, (3, 3), 1, 1, bias=True)
        self.conv9 = nn.Conv2d(32 + 80, 32, (3, 3), 1, 1, bias=True)
        self.conv10 = nn.Conv2d(32 + 1, 1, (3, 3), 1, 1, bias=True)

        self.num_layer_confidence_data = 36

        # Init Weights
        cc = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, \
              self.conv6, self.conv7, self.conv8, self.conv9, self.conv10, ]
        for m in cc:
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

        if isinstance(pretrained,nn.Module):
            pretrained_dict = pretrained.state_dict()
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)


    def forward(self, x0):

        x0_rgb = x0[:, :3, :, :]
        x0_d = x0[:, 3:4, :, :]

        if x0.shape[1] == 4:
            c0 = (x0_d > 0).float()
        else:
            c0 = x0[:, 4:5, :, :]


        # Depth Network
        xout_d, cout_d = self.d_net(x0_d, c0)


        # U-Net
        x1 = F.relu(self.conv1(torch.cat((xout_d, x0_rgb, cout_d), 1)))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))


        # Upsample 1
        x5u = F.interpolate(x5, x4.size()[2:], mode='nearest')
        x6 = F.leaky_relu(self.conv6(torch.cat((x5u, x4), 1)), 0.2)

        # Upsample 2
        x6u = F.interpolate(x6, x3.size()[2:], mode='nearest')
        x7 = F.leaky_relu(self.conv7(torch.cat((x6u, x3), 1)), 0.2)

        # Upsample 3
        x7u = F.interpolate(x7, x2.size()[2:], mode='nearest')
        x8 = F.leaky_relu(self.conv8(torch.cat((x7u, x2), 1)), 0.2)

        # Upsample 4
        x8u = F.interpolate(x8, x1.size()[2:], mode='nearest')
        x9 = F.leaky_relu(self.conv9(torch.cat((x8u, x1), 1)), 0.2)

        # Upsample 5
        x9u = F.interpolate(x9, x0_d.size()[2:], mode='nearest')
        last_layer_input = torch.cat((x9u, x0_d), 1)
        x10 = F.leaky_relu(self.conv10(last_layer_input), 0.2)
        layer_output = torch.cat((last_layer_input,xout_d, cout_d,x10), 1) # 32 + 2 + 1 + 1
        self.conf_features = layer_output
        return x10



class NconvMS(nn.Module):

    def __init__(self,pos_fn=None):
        super(NconvMS,self).__init__()

        self.out_feature_channels = 1

        # Import the unguided network
        self.d_net = unguided_net(pos_fn)

        self.d = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU()
        )  # 11,664 Params

        # RGB stream
        self.rgb = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU()
        )  # 186,624 Params

        # Fusion stream
        self.fuse = nn.Sequential(
            nn.Conv2d(80, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            nn.Conv2d(32, 1, 1, 1),
            # nn.Sigmoid()
        )  # 156,704 Params

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

        self.x0_d = []
        self.xout_d = []
        self.x0_rgb = []
        self.xout_rgb = []
        self.xf = []

        self.c0 = []
        self.cout_d = []

    def forward(self, x0, build_conf_input):

        assert x0.shape[1] == 4, "The input is not RGB-D"

        x0_rgb = x0[:, :3, :, :]
        x0_d = x0[:, 3:, :, :]
        c0 = (x0_d > 0).float()

        self.x0_d = x0_d
        self.x0_rgb = x0_rgb
        self.c0 = c0

        # Depth Network
        xout_d, cout_d = self.d_net(x0_d, c0)

        xout_d = self.d(xout_d)

        self.xout_d = xout_d
        self.cout_d = cout_d

        # RGB network
        xout_rgb = self.rgb(torch.cat((x0_rgb, cout_d), 1))
        self.xout_rgb = xout_rgb

        # Fusion Network
        xout = self.fuse(torch.cat((xout_rgb, xout_d), 1))

        self.xf = xout
        self.cf = cout_d

        return xout,cout_d




################################################
# criteria nets
################################################
class DualLoss(nn.Module):

    def __init__(self, net_a, net_b, weight_b):  # ged_train_weights
        super(DualLoss, self).__init__()
        self.net_a = net_a
        self.net_b = net_b
        self.weight_b = weight_b

    def forward(self, input, depth_a, depth_b, target_depth, epoch=None):
        error_a = self.net_a(input, depth_a, target_depth, epoch)
        self.loss = self.net_a.loss

        if self.weight_b > 0:
            error_b = self.net_b(input, depth_b, target_depth, epoch)
            self.loss[1] = self.net_b.loss[0]

            return (self.weight_b * error_b) + ((1.0 - self.weight_b) * error_a)

        return error_a


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self,depth_input, depth_prediction, depth_target,epoch=None):
        assert depth_prediction.dim() == depth_target.dim(), "inconsistent dimensions"

        valid_mask = ((depth_target>0).detach())

        num_valids = valid_mask.sum()
        assert (num_valids > 100), 'training image has less than 100 valid pixels'

        diff = depth_target - depth_prediction
        diff = diff[valid_mask]

        final_loss = (diff ** 2).mean()

        self.loss = [final_loss.item(),0,0]

        return final_loss

class InvertedMaskedL1Loss(nn.Module):
    def __init__(self):
        super(InvertedMaskedL1Loss, self).__init__()

    def forward(self,depth_input, depth_prediction, depth_target,epoch=None):

        assert depth_prediction.dim() == depth_target.dim(), "inconsistent dimensions"
        valid_mask = ((depth_target>0).detach())

        num_valids = valid_mask.sum()
        assert (num_valids > 100), 'training image has less than 100 valid pixels'

        diff = ((1.0/(depth_target[valid_mask])) - (1.0/(depth_prediction[valid_mask]))).abs()
        final_loss = diff.mean()

        self.loss = [final_loss.item(),0,0]

        return final_loss

class MaskedAbsRelLoss(nn.Module):
    def __init__(self):
        super(MaskedAbsRelLoss, self).__init__()

    def forward(self,depth_input, depth_prediction, depth_target,epoch=None):

        assert depth_prediction.dim() == depth_target.dim(), "inconsistent dimensions"

        valid_mask = ((depth_target>0).detach())

        num_valids = valid_mask.sum()
        assert (num_valids > 100), 'training image has less than 100 valid pixels'

        diff = depth_target - depth_prediction
        diff = diff[valid_mask]

        abs_rel_diff = (diff/depth_target[valid_mask]).abs()

        final_loss = abs_rel_diff.mean()

        self.loss = [final_loss.item(),0,0]

        return final_loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.loss = -1

    def forward(self,depth_input, depth_prediction, depth_target,epoch=None):
        assert depth_prediction.dim() == depth_target.dim(), "inconsistent dimensions"
        valid_mask = (depth_target>0).detach()

        num_valids = valid_mask.sum()
        assert (num_valids > 100), 'training image has less than 100 valid pixels'

        diff = depth_target - depth_prediction
        diff = diff[valid_mask]

        final_loss = diff.abs().mean()
        self.loss = [final_loss.item(),0,0]

        return final_loss
