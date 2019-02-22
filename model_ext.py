import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

def init_weights(m):
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

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

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

class DepthCompletionNet(nn.Module):

    def __init__(self, layers=18, modality_format='rgbd', pretrained=True):
        self.modality = modality_format
        if isinstance(pretrained,bool):
            self.create_from_zoo(layers=layers, pretrained=pretrained)
        elif isinstance(pretrained,DepthCompletionNet):
            self.load_from_depth_completion_net(pretrained)

    def create_from_zoo(self, layers=18,pretrained=True):

        assert(not 'w' in self.modality)

        assert ( layers in [18, 34, 50, 101, 152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(DepthCompletionNet, self).__init__()

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

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

    def load_from_depth_completion_net(self,pretrained_model):
        #assert(pretrained_model.version == 'dc_v1')
        super(DepthCompletionNet, self).__init__()
        self.version = 'dwc_v1'

        if 'dw' in self.modality:
            assert (False)  # dont make sense
        elif 'd' in self.modality:
            self.conv1_d = pretrained_model.conv1_d

        if 'rgb' in self.modality or 'g' in self.modality:
            self.conv1_img = pretrained_model.conv1_img

        self.conv2 = pretrained_model.conv2
        self.conv3 = pretrained_model.conv3
        self.conv4 = pretrained_model.conv4
        self.conv5 = pretrained_model.conv5
        self.conv6 = pretrained_model.conv6

        # decoding layers

        self.convt5 = pretrained_model.convt5
        self.convt4 = pretrained_model.convt4
        self.convt3 = pretrained_model.convt3
        self.convt2 = pretrained_model.convt2
        self.convt1 = pretrained_model.convt1
        self.convtf = pretrained_model.convtf

        del pretrained_model  # clear memory

    def forward(self, x):
        # print(x.shape)
        d = x[:,3, :, :].unsqueeze(1)
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
            return y
        else:
            return F.relu(y)
        return y


class DepthWeightCompletionNet(nn.Module):

    def create_from_zoo(self, layers=18,pretrained=True,dw_head_type='CBR'):
        self.version ='dwc_v1'
        if 'w' in self.modality:
            assert ( 'd' in self.modality )
        assert ( layers in [18, 34, 50, 101, 152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(DepthWeightCompletionNet, self).__init__()
        used_channels = 0

        if 'dw' in self.modality:
            channels = 64 // (len(self.modality) - 1)
            self.conv1_d = build_dw_head(2, channels,dw_head_type)
        elif 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

        if 'rgb' in self.modality:
            channels = 64 * 3 // (len(self.modality)-1 if 'w' in self.modality else len(self.modality))
            self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        elif 'g' in self.modality:
            channels = 64 // (len(self.modality)-1 if 'w' in self.modality else len(self.modality))
            self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)

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

    def __init__(self, layers=18, modality_format='rgbd', pretrained=True,dw_head_type='CBR'):
        self.modality = modality_format
        if isinstance(pretrained,bool):
            self.create_from_zoo(layers=layers, pretrained=pretrained,dw_head_type=dw_head_type)
        elif isinstance(pretrained,DepthCompletionNet):
            self.load_from_depth_completion_net(pretrained,dw_head_type=dw_head_type)

    def load_from_depth_completion_net(self,pretrained_model,dw_head_type):
        #assert(pretrained_model.version == 'dc_v1')
        super(DepthWeightCompletionNet, self).__init__()
        self.version = 'dwc_v1'

        if 'dw' in self.modality:
            channels = 64 // (len(self.modality) - 1)
            self.conv1_d = build_dw_head(2, channels,dw_head_type)
        else:
            assert(False) #dont make sense

        if 'rgb' in self.modality or 'g' in self.modality:
            self.conv1_img = pretrained_model.conv1_img

        self.conv2 = pretrained_model.conv2
        self.conv3 = pretrained_model.conv3
        self.conv4 = pretrained_model.conv4
        self.conv5 = pretrained_model.conv5
        self.conv6 = pretrained_model.conv6

        # decoding layers

        self.convt5 = pretrained_model.convt5
        self.convt4 = pretrained_model.convt4
        self.convt3 = pretrained_model.convt3
        self.convt2 = pretrained_model.convt2
        self.convt1 = pretrained_model.convt1
        self.convtf = pretrained_model.convtf

        del pretrained_model  # clear memory



    def forward(self, x):
        channel_offset = 0;
        # first layer
        if 'rgb' in self.modality:
            rgb = x[:, channel_offset:3, :, :]
            conv1_img = self.conv1_img(rgb)
            channel_offset = channel_offset +3
        elif 'g' in self.modality:
            g = x[:, channel_offset:1, :, :]
            conv1_img = self.conv1_img(g)
            channel_offset = channel_offset + 1

        if 'dw' in self.modality:
            d = x[:, channel_offset:(channel_offset+2), :, :]
            conv1_d = self.conv1_d(d)
        elif 'd' in self.modality:
            d = x[:, channel_offset:(channel_offset + 1), :, :]
            conv1_d = self.conv1_d(d)

        if 'rgbd' in self.modality or 'gd' in self.modality:
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
            return y
        else:
            return F.relu(y)
        return y
