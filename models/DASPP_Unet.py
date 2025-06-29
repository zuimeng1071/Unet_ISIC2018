import torch
import torch.nn as nn
from torch.nn import functional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1):
        super(DenseASPPConv, self).__init__()
        self.DenseASPPConv_block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate,
                      padding=atrous_rate, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.drop_rate = drop_rate

    def forward(self, input):
        x = self.DenseASPPConv_block(input)
        if self.drop_rate > 0:
            x = functional.dropout(x, p=self.drop_rate, training=self.training)
        return x


class DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, out_channels):
        super(DenseASPPBlock, self).__init__()
        self.aspp_3 = DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1)
        self.aspp_6 = DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1)
        self.aspp_12 = DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1)
        self.aspp_18 = DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1)
        self.aspp_24 = DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1)
        self.conv_1 = nn.Conv2d(in_channels + inter_channels2 * 5, out_channels, 1, 1)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)
        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)
        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)
        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)
        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        x = self.conv_1(x)
        return x


# 卷积块 用于提取特征
class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            # 3*3卷积块，填充1，步长2，填充模式为反射
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=1, padding_mode='reflect',
                      kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.PReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, padding=1, padding_mode='reflect',
                      kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.PReLU(),
        )

    def forward(self, input):
        # print(input.shape, self.block(input).shape, "Conv")
        return self.block(input)


# 下采样，使用卷积进行，图像缩小2倍
class DownSample(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Down_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, stride=2,
                      padding_mode='reflect', bias=False),  # 卷积
            nn.BatchNorm2d(channels),  # 批归一化
            nn.PReLU()
        )

    def forward(self, input):
        return self.Down_block(input)


# 上采样 使用插值法，并1*1卷积将通道数减半，进行上采样，图像扩大2倍
class UpSample(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv_1_1 = nn.Conv2d(channels, int(channels * 0.5), 1, stride=1, bias=False)

    def forward(self, input):
        # 最近邻插值法
        up_data = functional.interpolate(input, scale_factor=2, mode='nearest')
        conv_1 = self.Conv_1_1(up_data)
        # print(up_data.shape, conv_1.shape, input.shape)
        return conv_1


# Unet网络结构，包含编码器和解码器
class DASPP_Unet(nn.Module):
    """
    加入了DenseASPP的Unet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 编码器部分,使用DenseASPP
        self.c1 = DenseASPPBlock(3, 8, 16, 64)
        self.d1 = DownSample(64)
        self.c2 = DenseASPPBlock(64, 64, 64, 128)
        self.d2 = DownSample(128)
        self.c3 = DenseASPPBlock(128, 128, 128, 256)
        self.d3 = DownSample(256)
        self.c4 = DenseASPPBlock(256, 256, 256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_block(512, 1024)

        # 解码器部分
        self.u1 = UpSample(1024)
        self.c1_u = Conv_block(1024, 512)
        self.u2 = UpSample(512)
        self.c2_u = Conv_block(512, 256)
        self.u3 = UpSample(256)
        self.c3_u = Conv_block(256, 128)
        self.u4 = UpSample(128)
        self.c4_u = Conv_block(128, 64)
        self.c_out = nn.Conv2d(64, 1, 1, 1
                               , bias=False)
        # 激活函数，目前视为二分类，故用Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # 编码器
        S_1 = self.c1(input)
        S_2 = self.c2(self.d1(S_1))
        S_3 = self.c3(self.d2(S_2))
        S_4 = self.c4(self.d3(S_3))
        S_5 = self.c5(self.d4(S_4))

        # 解码器
        S_6 = self.c1_u(torch.cat((self.u1(S_5), S_4), dim=1))
        S_7 = self.c2_u(torch.cat((self.u2(S_6), S_3), dim=1))
        S_8 = self.c3_u(torch.cat((self.u3(S_7), S_2), dim=1))
        S_9 = self.c4_u(torch.cat((self.u4(S_8), S_1), dim=1))
        S_out = self.c_out(S_9)
        return self.sigmoid(S_out)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 创建模型
    net = DASPP_Unet().to(device)
    # 创建小规模测试数据
    x = torch.randn(1, 3, 384, 384, dtype=torch.float32, requires_grad=True).to(device)

    x.retain_grad()
    # 前向传播
    y = net(x)
    # 创建虚拟损失
    loss = (y - torch.randn_like(y)).pow(2).mean()
    # 反向传播
    loss.backward()

    # 检查梯度是否存在
    print("输入梯度是否存在:", x.grad is not None)
    print("参数量:", sum(p.numel() for p in net.parameters()))
    print("显存占用:", torch.cuda.memory_allocated() / 1024 ** 2, "MB")

    with torch.no_grad():
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = net(x)
        ender.record()
        torch.cuda.synchronize()
        print("推理时间:", starter.elapsed_time(ender), "ms")


if __name__ == "__main__":
    main()
