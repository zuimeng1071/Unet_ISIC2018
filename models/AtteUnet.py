import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch_out,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, convTranspose=True):
        super(up_conv, self).__init__()
        if convTranspose:
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_in, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2)

        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        x = self.Conv(x)
        return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    """
    in_channel: input image channels
    num_classes: output class number
    channel_list: a channel list for adjust the model size
    checkpoint: 是否有checkpoint  if False： call normal init
    convTranspose: 是否使用反卷积上采样。True: use nn.convTranspose  Flase: use nn.Upsample
    """

    def __init__(self,
                 in_channel=3,
                 num_classes=1,
                 channel_list=[64, 128, 256, 512, 1024],
                 checkpoint=False,
                 convTranspose=True):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channel, ch_out=channel_list[0])
        self.Conv2 = conv_block(ch_in=channel_list[0], ch_out=channel_list[1])
        self.Conv3 = conv_block(ch_in=channel_list[1], ch_out=channel_list[2])
        self.Conv4 = conv_block(ch_in=channel_list[2], ch_out=channel_list[3])
        self.Conv5 = conv_block(ch_in=channel_list[3], ch_out=channel_list[4])

        self.Up5 = up_conv(ch_in=channel_list[4], ch_out=channel_list[3], convTranspose=convTranspose)
        self.Att5 = Attention_block(F_g=channel_list[3],
                                    F_l=channel_list[3],
                                    F_int=channel_list[2])
        self.Up_conv5 = conv_block(ch_in=channel_list[4],
                                   ch_out=channel_list[3])

        self.Up4 = up_conv(ch_in=channel_list[3], ch_out=channel_list[2], convTranspose=convTranspose)
        self.Att4 = Attention_block(F_g=channel_list[2],
                                    F_l=channel_list[2],
                                    F_int=channel_list[1])
        self.Up_conv4 = conv_block(ch_in=channel_list[3],
                                   ch_out=channel_list[2])

        self.Up3 = up_conv(ch_in=channel_list[2], ch_out=channel_list[1], convTranspose=convTranspose)
        self.Att3 = Attention_block(F_g=channel_list[1],
                                    F_l=channel_list[1],
                                    F_int=64)
        self.Up_conv3 = conv_block(ch_in=channel_list[2],
                                   ch_out=channel_list[1])

        self.Up2 = up_conv(ch_in=channel_list[1], ch_out=channel_list[0], convTranspose=convTranspose)
        self.Att2 = Attention_block(F_g=channel_list[0],
                                    F_l=channel_list[0],
                                    F_int=channel_list[0] // 2)
        self.Up_conv2 = conv_block(ch_in=channel_list[1],
                                   ch_out=channel_list[0])

        self.Conv_1x1 = nn.Conv2d(channel_list[0],
                                  num_classes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

        if not checkpoint:
            init_weights(self)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoder
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoder
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1
