import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from model_ext import init_weights,conv_bn_relu
import inverse_warp as iw
import matplotlib.pyplot as plt

def build_no_grad_mask(depth):
    valid_mask = ((depth > 0).detach())
    mask = torch.zeros_like(depth)
    mask[valid_mask] = 1
    return mask


def conv_bn_relu_resnet(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True, resnet_layer=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    if resnet_layer:
        layers.append(resnet.BasicBlock(out_channels, out_channels))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


def conv_depth_features_validation(channels):

    layers = []
    layers.append(nn.Conv2d(channels, channels, 3, 1,
        1, bias=False))
    layers.append(nn.BatchNorm2d(channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(nn.Conv2d(channels, 1, 1, 1,
                            0, bias=True))

    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


def build_dw_head(in_channels, out_channels,type='CBR'):
    if type == 'CBR':
        head = conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        head.apply(init_weights)

    elif type == 'ResBlock1':
        head = conv_bn_relu_resnet(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    return head


def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


class SingleDepthCompletionNet(nn.Module):

    def __init__(self, layers=18, modality_format='rgbd', pretrained=True):
        self.modality = modality_format


        if not isinstance(pretrained,bool):
            pretrained_dict = pretrained.state_dict()
            use_resnet =True
        else:
            use_resnet = pretrained

        self.create_from_zoo(layers=layers, pretrained=use_resnet)

        if not isinstance(pretrained, bool):
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)

    def create_from_zoo(self, layers=18,pretrained=True):

        assert(not 'w' in self.modality)

        assert ( layers in [18, 34, 50, 101, 152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(SingleDepthCompletionNet, self).__init__()

        # if 'd' in self.modality:
        #     channels = 64 // len(self.modality)
        #     self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
        # if 'rgb' in self.modality:
        #     channels = 64 * 3 // len(self.modality)
        #     self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        # elif 'g' in self.modality:
        #     channels = 64 // len(self.modality)
        #     self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

        self.conv1_d = conv_bn_relu(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_img = conv_bn_relu(3, 32, kernel_size=3, stride=1, padding=1)
        pretrained_model = resnet.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if not pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model # clear memory

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512, out_channels=256,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768, out_channels=128,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256+128), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128+64), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128, out_channels=64,
            kernel_size=kernel_size, stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


    def forward(self, x):
        # print(x.shape)
        d = x[:,3:, :, :]
        rgb = x[:,:3, :, :]

        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(d)
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(rgb)
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality=='rgbd' or self.modality=='gd':
            conv1 = torch.cat((conv1_d, conv1_img),1)
        else:
            conv1 = conv1_d if (self.modality=='d') else conv1_img


        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2) # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3) # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4) # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5) # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        # print(conv5.shape)
        # print(convt5.shape)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1,conv1),1)

        y = self.convtf(y)

        if self.training:
            return y , convt1
        else:
            return F.relu(y), convt1
        return y


class EarlyFusionNet(nn.Module):

    def __init__(self, single_depth_completion_model):
        super(EarlyFusionNet, self).__init__()
        self.num_previous_features = 65 # photometric error + 64 features from the last deconvolution
        self.single_depth_completion_model = single_depth_completion_model
        #self.single_depth_completion_model.freeze()
        self.softmax2d = nn.Softmax2d()
        self.previous_confidence_stack = conv_depth_features_validation(self.num_previous_features)


    #curr_input (d+w from slam in the current time)
    def forward(self, curr_input,previous_projected_input=None):
        assert curr_input.shape[1] == 6, "current input with wrong size"
        if previous_projected_input is not None:
            assert previous_projected_input.shape[1] == self.num_previous_features+1, "previous_projected_input with wrong size"

        curr_input_depth = curr_input[:,3:4,:,:]
        curr_input_confidence = curr_input[:, 4:5, :, :]
        if previous_projected_input is not None:
            previous_proj_depth = previous_projected_input[:,0:1,:,:]
            previous_proj_confidence_features = previous_projected_input[:,1:(self.num_previous_features+2),:,:]
            previous_confidence = self.previous_confidence_stack(previous_proj_confidence_features)

            confidences = torch.cat([curr_input_confidence,previous_confidence],dim=1)
            weights = self.softmax2d(confidences)
            previous_weighted_depth = weights[:, 1:2, :, :] * previous_proj_depth
            previous_weighted_confidence = weights[:, 1:2, :, :] * previous_confidence

            curr_weighted_depth = weights[:,0:1,:,:]*curr_input_depth
            curr_weighted_confidence = weights[:, 0:1, :, :] * curr_input_confidence

            fused_depth = curr_weighted_depth + previous_weighted_depth
            fused_confidence = curr_weighted_confidence + previous_weighted_confidence
            fused_mask = build_no_grad_mask(fused_depth)
            single_input = torch.cat([ curr_input[:,0:3,:,:], fused_depth,fused_confidence ,fused_mask],dim=1)
        else:
            single_input = curr_input

        assert single_input.shape[1] == 6, "current input with wrong size"
        return self.single_depth_completion_model(single_input)


class LateFusionNet(nn.Module):

    def __init__(self, num_curr_features, num_previous_features):
        super(LateFusionNet, self).__init__()
        self.softmax2d = nn.Softmax2d()
        self.previous_confidence_stack = conv_depth_features_validation(num_previous_features)
        self.curr_confidence_stack = conv_depth_features_validation(num_curr_features)
        self.num_previous_features = num_previous_features
        self.num_curr_features = num_curr_features

    # curr_input (d+w from slam in the current time)
    def forward(self, curr_depth,curr_features, previous_projected_input=None):
        assert curr_depth.shape[1] == self.num_curr_features, "current input with wrong size"
        assert previous_projected_input.shape[
                   1] == self.num_previous_features, "previous_projected_input with wrong size"

        curr_input_depth = curr_input[:, 0:1, :, :]
        curr_input_confidence_features = curr_input[:, 1:(self.num_curr_features + 2), :, :]
        previous_proj_depth = curr_input[:, 0:1, :, :]
        previous_proj_confidence_features = curr_input[:, 1:(self.num_previous_features + 2), :, :]

        curr_confidence = self.curr_confidence_stack(previous_proj_confidence_features)
        previous_confidence = self.previous_confidence_stack(previous_proj_confidence_features)

        confidences = torch.cat([curr_confidence, previous_confidence], dim=1)
        weights = self.softmax2d(confidences)

        curr_weighted_depth = weights[:, 0, :, :] * curr_input_depth
        curr_weighted_confidence = weights[:, 0, :, :] * curr_confidence

        previous_weighted_depth = weights[:, 1, :, :] * previous_proj_depth
        previous_weighted_confidence = weights[:, 1, :, :] * previous_confidence

        fused_depth = curr_weighted_depth + previous_weighted_depth
        fused_confidence = curr_weighted_confidence + previous_weighted_confidence

        return fused_depth, fused_confidence

class ConfidenceNet(nn.Module):

    #curr: 64
    #previous: photometric error + depth error +(normals later)
    def __init__(self, num_curr_features, num_previous_features):
        super(ConfidenceNet, self).__init__()

        self.num_previous_features = num_previous_features
        self.num_curr_features = num_curr_features
        self.curr_confidence_stack = conv_depth_features_validation(self.num_previous_features+self.num_curr_features)

    # curr_input (d+w from slam in the current time)
    def forward(self, curr_features, previous_features):
        assert curr_features.shape[1] == self.num_curr_features, "current input with wrong size"
        assert previous_features.shape[
                   1] == self.num_previous_features, "previous_projected_input with wrong size"


        conf_features = torch.cat([curr_features,previous_features], dim=1)

        curr_confidence = self.curr_confidence_stack(conf_features)


        return curr_confidence


class LoopConfidenceNet(nn.Module):

    #curr: 64
    #previous: photometric error + depth error +(normals later)
    def __init__(self,single_dc_train_weights):
        super(LoopConfidenceNet, self).__init__()
        self.curr_confidence_stack = conv_depth_features_validation(64+2)
        self.singledc = SingleDepthCompletionNet(layers=18, modality_format='rgbd',
                                 pretrained=single_dc_train_weights)
        self.singledc.eval()

        self.singledc_2 = SingleDepthCompletionNet(layers=18, modality_format='rgbd',
                                                 pretrained=single_dc_train_weights)


    def getTrainableParameters(self):
        return list(self.curr_confidence_stack.parameters()) + list(self.singledc_2.parameters())

    # curr_input (d+w from slam in the current time)
    def forward(self, slam_features,previous_rgb=None,previous_depth_prediction=None,r_mat=None, t_vec=None,scale=None,intrinsics=None):


        depth_prediction1,depth_features1 = self.no_grad_depth_completion(slam_features)
        if previous_rgb is None:
            return depth_prediction1

        previous_features = self.no_grad_previous_features(depth_prediction1.clone(),previous_rgb,previous_depth_prediction,r_mat, t_vec,scale,intrinsics)

        conf_features = torch.cat([depth_features1, previous_features], dim=1)
        curr_confidence = torch.sigmoid(self.curr_confidence_stack(conf_features))

        slam_features[:,3:4,:,:] = depth_prediction1
        slam_features[:, 4:5, :, :] = curr_confidence

        depth_prediction2, _ = self.singledc_2(slam_features)

        return depth_prediction2

    def no_grad_depth_completion(self,slam_features):
        with torch.no_grad():
            return self.singledc(slam_features)

    def no_grad_previous_features(self,curr_depth,previous_rgb,previous_depth_prediction,r_mat, t_vec,scale,intrinsics):
        with torch.no_grad():
            for cb in range(curr_depth.shape[0]):
                curr_depth[cb, 0, :, :] /= scale[cb]
            rgb_projected_back, depth_error = iw.homography_from_and_depth(previous_rgb,previous_depth_prediction, curr_depth,
                                                              r_mat, t_vec,
                                                              intrinsics.scale(previous_rgb.shape[2], previous_rgb.shape[3]).cuda())
            photometric_error = (rgb_projected_back - previous_rgb).abs().norm(dim=1, keepdim=True)

            # im = depth_error[0,0,:,:].cpu().numpy()
            # #img1[0, :, :] = rgb2grayscale(img1) * 5
            # # img1[1, :, :] = rgb2grayscale(img2)*5
            # # img1[2, :, :] = 0
            # imgplot = plt.imshow(im)
            # plt.show()

            for cb in range(curr_depth.shape[0]):
                depth_error[cb, 0, :, :] *= scale[cb]

            features_vec = torch.cat([photometric_error, depth_error], dim=1)

        return features_vec


from nconv_sd import CNN as unguided_net


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
        layer_output = torch.cat((last_layer_input,xout_d, cout_d,x10), 1) # 33 + 1 + 1 + 1
        self.conf_features = layer_output
        return x10


class SingleFrameConfidenceNet(nn.Module):


    def __init__(self, singledcnet): #ged_train_weights
        super(SingleFrameConfidenceNet, self).__init__()
        self.curr_confidence_stack = conv_depth_features_validation(singledcnet.num_layer_confidence_data)
        self.singledc = singledcnet
        self.train_mode()

    def getTrainableParameters(self,discriminator_only = True):
        if discriminator_only:
            return list(self.curr_confidence_stack.parameters())
        return self.parameters()

    def train_mode(self,discriminator_only = True):
        self.curr_confidence_stack.train()
        if discriminator_only:
            self.singledc.eval()

    def eval_mode(self):
        self.curr_confidence_stack.eval()
        self.singledc.eval()

    def forward(self, rgbd):

        depth = self.singledc(rgbd)
        conf_features = self.singledc.conf_features
        curr_confidence = torch.sigmoid(self.curr_confidence_stack(conf_features))

        return torch.cat((depth,curr_confidence), 1)

import copy

class NConvLoss(nn.Module):

    def __init__(self, singledcnet,lossnet):  # ged_train_weights
        super(NConvLoss, self).__init__()

        self.singledc = singledcnet
        self.singledc.eval()
        self.loss_fn = lossnet
        self.loss_fn2 = copy.deepcopy(lossnet)

    def forward(self,input,pred, target_depth,epoch):

        rgbdc = torch.cat([input[:,:3,:,:],pred], dim=1)
        input_confidence = ((input[:,3:4,:,:]>0).detach()).float()
        new_pred = self.singledc(rgbdc)
        self.new_prediction = new_pred
        result_depth = self.loss_fn(new_pred[:, :1, :, :], target_depth)
        self.loss = self.loss_fn.loss

        result_conf = self.loss_fn2(pred[:, 1:2, :, :], input_confidence)
        self.loss[2] = self.loss_fn2.loss[0]

        result_depth_old = self.loss_fn(pred[:, :1, :, :], target_depth)
        self.loss[1] = self.loss_fn.loss[0]

        return result_depth + result_conf

class NConvLoss2(nn.Module):

    def __init__(self, singledcnet,lossnet):  # ged_train_weights
        super(NConvLoss2, self).__init__()
        self.singledc = singledcnet
        self.singledc.eval()
        self.loss_fn = lossnet

    def forward(self,input,pred, target_depth,epoch):
        rgbdc = torch.cat([input[:,:3,:,:],pred], dim=1)
        new_pred = self.singledc(rgbdc)
        self.new_prediction = new_pred
        result_depth = self.loss_fn(new_pred[:, :1, :, :], target_depth)
        self.loss = self.loss_fn.loss
        result_depth_old = self.loss_fn(pred[:, :1, :, :], target_depth)
        self.loss[1] = self.loss_fn.loss[0]
        return result_depth