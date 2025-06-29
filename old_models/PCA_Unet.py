import torch
import torch.nn as nn
from torch.nn import functional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.preferred_linalg_library('default')


def PCA(data: torch.Tensor, k=1):
    # 对所有特征进行中心化
    # n个样本， c个特征，降维至k个特征
    # [n, c]
    data_mean = data.mean(dim=0, keepdim=False)
    data_mean = data_mean.repeat(data.size(0), 1)
    data = data - data_mean
    # 求协方差矩阵
    CovMat = torch.cov(data.T, correction=1)
    # 求协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eig(CovMat)  # c, [c, c]
    # 求前k个大的特征值向量
    index = torch.argsort(eigenvalues.real, descending=True)
    index = index[:k]
    picked_eigenvectors = eigenvectors.real[:, index]
    # [c, k]
    return torch.matmul(data, picked_eigenvectors)


class PCA_Block(nn.Module):
    """
    PCA层
    """

    def __init__(self):
        super(PCA_Block, self).__init__()
        self.PCA = PCA

    def forward(self, data):
        # [batch_size, channels, x_size, y_size]
        batch_size, channels, x_size, y_size = data.shape
        k = channels // 2
        assert channels % 2 == 0
        # 维度变换
        data = data.permute(0, 2, 3, 1).reshape(-1, channels)  # [batch, x, y, c] -> [n, c]
        data = self.PCA(data, k=k)
        data = data.view(batch_size, x_size, y_size, k).permute(0, 3, 1, 2).contiguous()
        return data


# 卷积块 用于提取特征
class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, Dropout=0.3, *args, **kwargs):
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


# 下采样，使用卷积进行，图像缩小2倍
class DownSample(nn.Module):
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
        # print(input.shape, self.Down_block(input).shape, "Down")
        return self.Down_block(input)


# 上采样 使用插值法，并1*1卷积将通道数减半，进行上采样，图像扩大2倍
class UpSample(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv_1_1 = nn.Conv2d(channels, int(channels * 0.75), 1, stride=1, bias=False)

    def forward(self, input):
        # 最近邻插值法
        up_data = functional.interpolate(input, scale_factor=2, mode='nearest')
        conv_1 = self.Conv_1_1(up_data)
        # print(up_data.shape, conv_1.shape, input.shape)
        return conv_1


# Unet网络结构，包含编码器和解码器
class PCA_Unet(nn.Module):
    """
    在跳跃连接部分加入了PCA的Unet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        # 跳跃连接部分
        self.PCA = PCA_Block()

        # 解码器部分
        self.u1 = UpSample(1024)
        # 对c1输出的特征图1*1卷积， 用于跳跃连接
        self.c1_u = Conv_block(1024, 512)
        self.u2 = UpSample(512)
        # 对c2输出的特征图1*1卷积， 用于跳跃连接
        self.c2_u = Conv_block(512, 256)
        self.u3 = UpSample(256)
        # 对c3输出的特征图1*1卷积， 用于跳跃连接
        self.c3_u = Conv_block(256, 128)
        self.u4 = UpSample(128)
        # 对c4输出的特征图1*1卷积， 用于跳跃连接
        self.c4_u = Conv_block(128, 64)
        self.c_out = nn.Conv2d(64, 1, 1, 1
                               , bias=False)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        S_1 = self.c1(input)
        S_2 = self.c2(self.d1(S_1))
        S_3 = self.c3(self.d2(S_2))
        S_4 = self.c4(self.d3(S_3))
        S_5 = self.c5(self.d4(S_4))

        # PCA+S
        S_1 = self.PCA(S_1)
        S_2 = self.PCA(S_2)
        S_3 = self.PCA(S_3)
        S_4 = self.PCA(S_4)

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
    net = PCA_Unet().to(device)
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
