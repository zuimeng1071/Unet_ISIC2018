import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn.utils.parametrizations import orthogonal  # 引入正交参数化


"""
该代码实现了一个改进版的U-Net模型（），结合了以下特点：


"""


class OrthogonalLinear(nn.Module):
    """
    正交特征提取器
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用参数化正交层
        self.fc = orthogonal(nn.Linear(in_channels, out_channels, bias=False))

    def forward(self, x):
        B, C_in, H, W = x.shape
        x_centered = x - x.mean(dim=(2, 3), keepdim=True)  # 沿空间维度计算均值
        x_reshaped = x_centered.view(B * H * W, C_in)
        y_reshaped = self.fc(x_reshaped)
        y = y_reshaped.view(B, H, W, -1).permute(0, 3, 1, 2)
        return y


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze())
        max_out = self.fc(self.max_pool(x).squeeze())
        return (avg_out + max_out).unsqueeze(-1).unsqueeze(-1)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_att = ChannelAttention(channels)
        self.fusion_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, conv_feat):
        # 通道注意力加权
        conv_weight = self.conv_att(conv_feat)
        # 特征融合
        return self.fusion_conv(conv_feat * conv_weight)


class DenseASPPConv(nn.Sequential):
    """
    DenseASPP卷积块

    通过空洞卷积（Atrous Convolution）提取多尺度特征，同时加入Dropout防止过拟合。
    该模块用于构建DenseASPPBlock，逐步增加感受野。

    参数：
    - in_channels: 输入特征的通道数
    - inter_channels: 中间层的通道数
    - out_channels: 输出特征的通道数
    - atrous_rate: 空洞卷积的膨胀率
    - drop_rate: Dropout的概率
    """
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
    """
    DenseASPP块

    结合多个DenseASPPConv模块，逐步增加感受野并融合多尺度特征。
    通过级联方式将不同尺度的特征拼接在一起，增强特征表达能力。

    参数：
    - in_channels: 输入特征的通道数
    - inter_channels1: 第一中间层的通道数
    - inter_channels2: 第二中间层的通道数
    - out_channels: 输出特征的通道数
    """
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


class Conv_block(nn.Module):
    """
    标准卷积块

    包含两个3x3卷积层，每个卷积层后接批归一化、Dropout和激活函数（PReLU）。
    用于提取局部特征。

    参数：
    - in_channels: 输入特征的通道数
    - out_channels: 输出特征的通道数
    - Dropout: Dropout的概率
    """
    def __init__(self, in_channels, out_channels, Dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            # 3*3卷积块，填充1，步长1，填充模式为反射
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=1, padding_mode='reflect',
                      kernel_size=3, bias=False),
            # 批归一化
            nn.BatchNorm2d(out_channels),
            # Dropout
            nn.Dropout(Dropout),
            # 激活函数
            nn.PReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, padding=1, padding_mode='reflect',
                      kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(Dropout),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.block(input)


class DownSample(nn.Module):
    """
    下采样模块

    通过3x3卷积（步长为2）实现特征图的空间分辨率减半，同时保持通道数不变。
    用于编码器部分的特征压缩。

    参数：
    - channels: 输入特征的通道数
    """
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Down_block = nn.Sequential(
            # 一个3*3卷积。padding=1， stride = 2：保证卷积后特征图大小减半
            nn.Conv2d(channels, channels, 3, padding=1, stride=2,
                      padding_mode='reflect', bias=False),  # 卷积
            nn.BatchNorm2d(channels),  # 批归一化
            nn.PReLU()
        )

    def forward(self, input):
        return self.Down_block(input)


class UpSample(nn.Module):
    """
    上采样模块

    通过最近邻插值法放大特征图的空间分辨率，并使用正交投影层（OrthogonalConv1x1）减少通道数。
    用于解码器部分的特征恢复。

    参数：
    - channels: 输入特征的通道数
    """
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv_1_1 = OrthogonalLinear(channels, channels // 2)

    def forward(self, input):
        # 最近邻插值法
        up_data = functional.interpolate(input, scale_factor=2, mode='nearest')
        conv_1 = self.Conv_1_1(up_data)
        return conv_1


class DASPP_OProj_ChannelAtte_UNet(nn.Module):
    """
    改进后的Unet：加入可训练正交投影层和特征金字塔提取特征，并通过正交投影层实现类似PCA的主成分提取
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

        # 通道注意力
        self.attention_1 = Attention(64)
        self.attention_2 = Attention(128)
        self.attention_3 = Attention(256)
        self.attention_4 = Attention(512)

        # 解码器部分
        self.u1 = UpSample(1024)
        self.c1_u = Conv_block(1024, 512)
        self.u2 = UpSample(512)
        self.c2_u = Conv_block(512, 256)
        self.u3 = UpSample(256)
        self.c3_u = Conv_block(256, 128)
        self.u4 = UpSample(128)
        self.c4_u = Conv_block(128, 64)
        self.c_out = nn.Conv2d(64, 1, 1, 1, bias=False)
        # 激活函数，目前视为二分类，故用Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        S_1 = self.c1(input)
        S_2 = self.c2(self.d1(S_1))
        S_3 = self.c3(self.d2(S_2))
        S_4 = self.c4(self.d3(S_3))
        S_5 = self.c5(self.d4(S_4))

        A_1 = self.attention_1(S_1)
        A_2 = self.attention_2(S_2)
        A_3 = self.attention_3(S_3)
        A_4 = self.attention_4(S_4)

        # 解码器
        S_6 = self.c1_u(torch.cat((self.u1(S_5), A_4), dim=1))
        S_7 = self.c2_u(torch.cat((self.u2(S_6), A_3), dim=1))
        S_8 = self.c3_u(torch.cat((self.u3(S_7), A_2), dim=1))
        S_9 = self.c4_u(torch.cat((self.u4(S_8), A_1), dim=1))
        S_out = self.c_out(S_9)
        return self.sigmoid(S_out)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 创建模型
    net = DASPP_OProj_ChannelAtte_UNet().to(device)
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
