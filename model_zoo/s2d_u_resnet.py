import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from model_zoo.nconv_sd import CNN as unguided_net


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


class S2DUResNet(nn.Module):

    def __init__(self, layers=18, in_channels=3, out_channels=1, pretrained=True,unguided=False):

        self.out_channels = out_channels
        self.in_channels = in_channels

        if out_channels == 2:
            self.out_feature_channels = 1
        else:
            self.out_feature_channels = 129



        assert (layers in [18, 34, 50, 101,
                           152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(S2DUResNet, self).__init__()
        used_channels = 0

        # Import the unguided network
        self.unguided = unguided
        extra_channel = 0
        if unguided:
            self.unguidednet = unguided_net()
            extra_channel = 1

        if in_channels == 3: # rgb
            self.conv1_img = conv_bn_relu(3, 64, kernel_size=3, stride=1, padding=1)
        if in_channels == 4:  # rgbd
            self.conv1_img = conv_bn_relu(3, 48, kernel_size=3, stride=1, padding=1)
            self.conv1_d = conv_bn_relu(1+extra_channel, 16, kernel_size=3, stride=1, padding=1)
        if in_channels == 5:  # rgbd
            self.conv1_img = conv_bn_relu(3, 48, kernel_size=3, stride=1, padding=1)
            self.conv1_d = conv_bn_relu(2, 16, kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if not pretrained:
            pretrained_model.apply(init_weights)
        # self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

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
        self.convt3 = convt_bn_relu(in_channels=(256 + 128), out_channels=64,
                                    kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64), out_channels=64,
                                    kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128, out_channels=64,
                                    kernel_size=kernel_size, stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=self.out_channels, kernel_size=1, stride=1, bn=False, relu=False)


    def forward(self, x, build_conf_input):
        # first layer
        if self.in_channels == 3:
            conv1 = self.conv1_img(x[:, :3, :, :])
        if self.in_channels == 4:
            conv1_img = self.conv1_img(x[:, :3, :, :])
            if self.unguided:
                densex,coutx = self.unguidednet(x[:, 3:4, :, :],(x[:, 3:4, :, :]>0).float())
                conv1_d = self.conv1_d(torch.cat((densex, coutx), 1))
            else:
                conv1_d = self.conv1_d(x[:, 3:4, :, :])
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        if self.in_channels == 5:
            conv1_img = self.conv1_img(x[:, :3, :, :])
            if self.unguided:
                densex,coutx = self.unguidednet(x[:, 3:4, :, :], x[:, 4:5, :, :])
                conv1_d = self.conv1_d(torch.cat((densex, coutx), 1))
            else:
                conv1_d = self.conv1_d(x[:, 3:5, :, :])
            conv1 = torch.cat((conv1_d, conv1_img), 1)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        last_layer_input = torch.cat((convt1, conv1), 1)

        y = self.convtf(last_layer_input)

        if build_conf_input:
            if self.out_channels == 1:
                features = torch.cat((last_layer_input, y), 1)  # 129
            else:  # out_channels == 2
                features = torch.sigmoid(y[:, 1:2, :, :])  # it is already the confidence
        else:
            features = None

        if self.training:
            depth = y[:, 0:1, :, :]
        else:
            depth = F.relu(y[:, 0:1, :, :])

        return depth, features