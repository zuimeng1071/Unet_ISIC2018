import torch
import torch.nn as nn
from torch.nn import functional

"""
该代码实现了一个改进版的U-Net模型：
1. 正交投影层：通过正交约束的1x1卷积层（OrthogonalConv1x1）实现类似PCA的主成分分析效果。
   - 正交约束保证投影方向正交，减少特征冗余。
   - 输入数据中心化进一步增强了正交投影的效果。
PCA是投影后使每个成分之间的方差最大化，这里通过对卷积核参数施加正交约束，实现类似PCA的效果。
正交约束确保变换后的特征通道之间互不冗余，从而达到降维或解耦的目的。
"""


class OrthogonalLinear(nn.Module):
    """
    正交特征提取器

    通过正交约束的线性变换实现类PCA的特征投影，主要应用于特征空间的正交化降维。

    参数：
    - in_channels: 输入特征通道数
    - out_channels: 输出特征通道数（需小于 in_channels）

    输入形状：(B, C_in, H, W)
    输出形状：(B, C_out, H, W)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 初始化权重矩阵
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.orthogonal_(self.weight)  # 使用正交初始化

    def forward(self, x):
        # 原始输入形状: (Batch, Channels_in, Height, Width)
        B, C_in, H, W = x.shape

        # 空间维度中心化（沿H,W计算均值）
        x_centered = x - x.mean(dim=(2, 3), keepdim=True)

        # 重塑为线性层可处理的形式：(B*H*W, C_in)
        x_reshaped = x_centered.view(B * H * W, C_in)

        # 应用正交投影：(B*H*W, C_in) → (B*H*W, C_out)
        y_reshaped = torch.matmul(x_reshaped, self.weight.t())

        # 恢复空间结构并调整通道顺序
        # (B, C_out, H, W)
        y = y_reshaped.view(B, H, W, -1).transpose(1, 3)
        return y


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


class OrthoProj_UNet(nn.Module):
    """
    改进后的Unet：加入可训练正交投影层,通过正交投影层实现类似PCA的主成分提取
    """

    def __init__(self):
        super().__init__()
        # 编码器部分
        self.c1 = Conv_block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_block(512, 1024)

        # 解码器部分
        self.u1 = UpSample(1024)
        self.c1_u = Conv_block(1024, 512)  # 调整输入通道
        self.u2 = UpSample(512)
        self.c2_u = Conv_block(512, 256)
        self.u3 = UpSample(256)
        self.c3_u = Conv_block(256, 128)
        self.u4 = UpSample(128)
        self.c4_u = Conv_block(128, 64)
        self.c_out = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        S_1 = self.c1(input)
        S_2 = self.c2(self.d1(S_1))
        S_3 = self.c3(self.d2(S_2))
        S_4 = self.c4(self.d3(S_3))
        S_5 = self.c5(self.d4(S_4))

        # print(S_4.shape, self.u1(S_5).shape)
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
    net = OrthoProj_UNet().to(device)
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
