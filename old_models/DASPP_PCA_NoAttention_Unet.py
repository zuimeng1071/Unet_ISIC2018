import torch
import torch.nn as nn
from torch.nn import functional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.preferred_linalg_library('default')


def PCA(data: torch.Tensor, k=1):
    """主成分分析降维函数
    Args:
        data: 输入数据 [n_samples, n_features]
        k: 保留的主成分数量
    Returns:
        降维后的数据 [n_samples, k]
    """
    data_mean = data.mean(dim=0, keepdim=True)
    data_centered = data - data_mean  # 数据中心化
    U, S, Vt = torch.linalg.svd(data_centered, full_matrices=False)  # 奇异值分解
    Vk = Vt[:k, :].T  # 取前k个主成分
    return torch.matmul(data_centered, Vk)  # 投影到主成分空间


class PCA_Block(nn.Module):
    """
    PCA降维模块
    输入形状: [batch_size, channels, height, width]
    输出形状: [batch_size, k, height, width] (k = channels//2)
    """

    def __init__(self, channels):
        super(PCA_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.PCA = PCA
        self.bn2 = nn.BatchNorm2d(num_features=channels // 2)

    def forward(self, data):
        # 输入形状: [batch, c, h, w]
        batch_size, channels, x_size, y_size = data.shape
        k = channels // 2
        assert channels % 2 == 0

        # 检查NaN/inf
        if torch.isnan(data).any() or torch.isinf(data).any():
            print("输入存在 NaN/inf")
            data = torch.nan_to_num(data)  # 临时修复

        # BatchNorm
        data = self.bn1(data)
        # 维度变换 [batch, c, h, w] -> [batch*h*w, c]
        data = data.permute(0, 2, 3, 1).reshape(-1, channels)
        data = self.PCA(data, k=k)  # PCA降维到k维
        # 恢复形状 [batch, h, w, k] -> [batch, k, h, w]
        data = data.view(batch_size, x_size, y_size, k).permute(0, 3, 1, 2).contiguous()
        # data = self.bn2(data)

        return data


class PCA_Conv_Cat(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, conv_feat, pca_feat):
        # 特征融合
        fused = torch.cat([
            pca_feat,
            conv_feat
        ], dim=1)
        print(f"{torch.max(self.fusion_conv(fused))}")
        return self.fusion_conv(fused)


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


# 上采样 使用插值法，并1*1卷积将通道数*0.5，进行上采样，图像扩大2倍
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
class DASPP_PCA_NoAtten_Unet(nn.Module):

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

        # 跳跃连接处理
        # PCA降维模块
        self.PCA_1 = PCA_Block(64)
        self.PCA_2 = PCA_Block(128)
        self.PCA_3 = PCA_Block(256)
        self.PCA_4 = PCA_Block(512)

        # 通道调整卷积
        self.conv_S_1 = nn.Conv2d(64, 32, 1)  # 64 -> 32
        self.conv_S_2 = nn.Conv2d(128, 64, 1)  # 128->64
        self.conv_S_3 = nn.Conv2d(256, 128, 1)  # 256->128
        self.conv_S_4 = nn.Conv2d(512, 256, 1)  # 512->256
        # PCA与卷积特征融合
        self.PC_1 = PCA_Conv_Cat(32)
        self.PC_2 = PCA_Conv_Cat(64)
        self.PC_3 = PCA_Conv_Cat(128)
        self.PC_4 = PCA_Conv_Cat(256)

        # 解码器部分 (上采样路径)
        self.u1 = UpSample(1024)  # 输出 [b,512,48,48]
        self.c1_u = Conv_block(512 + 256, 512)  # 输入 [b,512+256,48,48] -> [b,512,48,48]
        self.u2 = UpSample(512)  # 输出 [b,256,96,96]
        self.c2_u = Conv_block(256 + 128, 256)  # 输入 [b,256+128,96,96] -> [b,256,96,96]
        self.u3 = UpSample(256)  # 输出 [b,128,192,192]
        self.c3_u = Conv_block(128 + 64, 128)  # 输入 [b,128+64,192,192] -> [b,128,192,192]
        self.u4 = UpSample(128)  # 输出 [b,64,384,384]
        self.c4_u = Conv_block(64 + 32, 64)  # 输入 [b,64+32,384,384] -> [b,64,384,384]
        self.c_out = nn.Conv2d(64, 1, 1)  # 输出 [b,1,384,384]
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # 初始输入检查
        if torch.isnan(input).any() or torch.isinf(input).any():
            print("Input contains NaN or Inf values!")

        S_1 = self.c1(input)
        S_2 = self.c2(self.d1(S_1))
        S_3 = self.c3(self.d2(S_2))
        S_4 = self.c4(self.d3(S_3))
        S_5 = self.c5(self.d4(S_4))

        # PCA+Conv_1 * 1 结合
        P_1 = self.PCA_1(S_1)
        P_2 = self.PCA_2(S_2)
        P_3 = self.PCA_3(S_3)
        P_4 = self.PCA_4(S_4)
        # 调整原始特征通道数并与PCA拼接
        S_1 = self.PC_1(self.conv_S_1(S_1), P_1)
        S_2 = self.PC_2(self.conv_S_2(S_2), P_2)
        S_3 = self.PC_3(self.conv_S_3(S_3), P_3)
        S_4 = self.PC_4(self.conv_S_4(S_4), P_4)

        # print(S_4.shape, self.u1(S_5).shape)
        S_6 = self.c1_u(torch.cat((self.u1(S_5), S_4), dim=1))
        S_7 = self.c2_u(torch.cat((self.u2(S_6), S_3), dim=1))
        S_8 = self.c3_u(torch.cat((self.u3(S_7), S_2), dim=1))
        S_9 = self.c4_u(torch.cat((self.u4(S_8), S_1), dim=1))
        S_out = self.c_out(S_9)
        return self.sigmoid(S_out)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        'Total params': total_params,
        'Trainable params': trainable_params,
        'Non-trainable params': non_trainable_params
    }


def print_layer_wise_parameters(model):
    for name, param in model.named_parameters():
        print(f'Layer: {name}, Parameters: {param.numel()}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 创建小规模测试数据，并转换为双精度浮点数
    x = torch.randn(1, 3, 384, 384, dtype=torch.float32, requires_grad=True).to(device)
    x.retain_grad()

    # 创建模型，并将模型参数转换为双精度浮点数
    net = PCA_Unet().to(device)

    # 前向传播
    y = net(x)

    # 创建虚拟损失
    loss = (y - torch.randn_like(y)).pow(2).mean()

    # 反向传播
    loss.backward()

    # 检查梯度是否存在
    print("输入梯度是否存在:", x.grad is not None)

    # 检查各层梯度
    for name, param in net.named_parameters():
        print(f"参数层: {name}, 梯度存在: {param.grad is not None}")
        pass

    # 打印每层参数量
    print_layer_wise_parameters(net)


if __name__ == "__main__":
    main()
