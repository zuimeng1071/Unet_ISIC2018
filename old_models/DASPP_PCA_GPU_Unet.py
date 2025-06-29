import torch
import torch.nn as nn
from torch.nn import functional

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

        return data


class ChannelAttention(nn.Module):
    """通道注意力机制
    输入形状: [batch, channels, height, width]
    输出形状: [batch, channels, 1, 1]
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 注意力生成网络
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze())  # 平均池化路径
        max_out = self.fc(self.max_pool(x).squeeze())  # 最大池化路径
        return (avg_out + max_out).unsqueeze(-1).unsqueeze(-1)  # 合并并恢复维度


class PCAAttention(nn.Module):
    """PCA与卷积特征融合模块
    输入形状:
        pca_feat: [batch, c, h, w]
        conv_feat: [batch, c, h, w]
    输出形状: [batch, c, h, w]
    """

    def __init__(self, channels):
        super().__init__()
        self.pca_att = ChannelAttention(channels)  # PCA特征注意力
        self.conv_att = ChannelAttention(channels)  # 卷积特征注意力
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1)  # 特征融合卷积

    def forward(self, pca_feat, conv_feat):
        # 计算注意力权重
        pca_weight = self.pca_att(pca_feat)
        conv_weight = self.conv_att(conv_feat)

        # 特征加权融合
        fused = torch.cat([pca_feat * pca_weight, conv_feat * conv_weight], dim=1)
        return self.fusion_conv(fused)


class DenseASPPConv(nn.Sequential):
    """密集ASPP卷积块
    输入形状: [batch, in_channels, h, w]
    输出形状: [batch, out_channels, h, w]
    """

    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate, drop_rate=0.1):
        super(DenseASPPConv, self).__init__()
        self.DenseASPPConv_block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),  # 1x1卷积降维
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate,  # 空洞卷积
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
    """密集ASPP模块
    输入形状: [batch, in_channels, h, w]
    输出形状: [batch, out_channels, h, w]
    """

    def __init__(self, in_channels, inter_channels1, inter_channels2, out_channels):
        super(DenseASPPBlock, self).__init__()
        # 多尺度空洞卷积层(3,6,12,18,24)
        self.aspp_3 = DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1)
        self.aspp_6 = DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1)
        self.aspp_12 = DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1)
        self.aspp_18 = DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1)
        self.aspp_24 = DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1)
        self.conv_1 = nn.Conv2d(in_channels + inter_channels2 * 5, out_channels, 1, 1)  # 最终融合卷积

    def forward(self, x):
        # 级联多尺度特征
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
        return self.conv_1(x)


class Conv_block(nn.Module):
    """双卷积块
    输入形状: [batch, in_channels, h, w]
    输出形状: [batch, out_channels, h, w]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.block(input)


class DownSample(nn.Module):
    """下采样模块(步长2卷积)
    输入形状: [batch, channels, h, w]
    输出形状: [batch, channels, h//2, w//2]
    """

    def __init__(self, channels):
        super().__init__()
        self.Down_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

    def forward(self, input):
        return self.Down_block(input)


class UpSample(nn.Module):
    """上采样模块(最近邻插值+1x1卷积)
    输入形状: [batch, channels, h, w]
    输出形状: [batch, channels*0.5, h*2, w*2]
    """

    def __init__(self, channels):
        super().__init__()
        self.Conv_1_1 = nn.Conv2d(channels, int(channels * 0.5), 1, bias=False)

    def forward(self, input):
        up_data = functional.interpolate(input, scale_factor=2, mode='nearest')  # 上采样
        return self.Conv_1_1(up_data)  # 通道调整


class DASPP_PCA_Unet(nn.Module):
    """基于DenseASPP和PCA注意力机制的Unet网络
    输入形状: [batch, 3, 384, 384]
    输出形状: [batch, 1, 384, 384]
    """

    def __init__(self):
        super().__init__()
        # 编码器部分 (下采样路径)
        self.c1 = DenseASPPBlock(3, 8, 16, 64)  # 输出 [b,64,384,384]
        self.d1 = DownSample(64)  # 输出 [b,64,192,192]
        self.c2 = DenseASPPBlock(64, 64, 64, 128)  # 输出 [b,128,192,192]
        self.d2 = DownSample(128)  # 输出 [b,128,96,96]
        self.c3 = DenseASPPBlock(128, 128, 128, 256)  # 输出 [b,256,96,96]
        self.d3 = DownSample(256)  # 输出 [b,256,48,48]
        self.c4 = DenseASPPBlock(256, 256, 256, 512)  # 输出 [b,512,48,48]
        self.d4 = DownSample(512)  # 输出 [b,512,24,24]
        self.c5 = Conv_block(512, 1024)  # 输出 [b,1024,24,24]

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
        # PCA与卷积特征注意力融合模块
        self.attn_1 = PCAAttention(32)
        self.attn_2 = PCAAttention(64)
        self.attn_3 = PCAAttention(128)
        self.attn_4 = PCAAttention(256)

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
        # 编码器前向传播
        S_1 = self.c1(input)  # [b,64,384,384]
        S_2 = self.c2(self.d1(S_1))  # [b,128,192,192]
        S_3 = self.c3(self.d2(S_2))  # [b,256,96,96]
        S_4 = self.c4(self.d3(S_3))  # [b,512,48,48]
        S_5 = self.c5(self.d4(S_4))  # [b,1024,24,24]

        # PCA特征提取与注意力融合
        P_1 = self.PCA_1(S_1)  # [b,32,384,384]
        P_2 = self.PCA_2(S_2)  # [b,64,192,192]
        P_3 = self.PCA_3(S_3)  # [b,128,96,96]
        P_4 = self.PCA_4(S_4)  # [b,256,48,48]
        # 调整原始特征通道并与PCA特征融合
        S_1 = self.attn_1(self.conv_S_1(S_1), P_1)  # [b,32,384,384]
        S_2 = self.attn_2(self.conv_S_2(S_2), P_2)  # [b,64,192,192]
        S_3 = self.attn_3(self.conv_S_3(S_3), P_3)  # [b,128,96,96]
        S_4 = self.attn_4(self.conv_S_4(S_4), P_4)  # [b,256,48,48]

        # 解码器前向传播(包含跳跃连接)
        S_6 = self.c1_u(torch.cat((self.u1(S_5), S_4), dim=1))  # [b,512,48,48]
        S_7 = self.c2_u(torch.cat((self.u2(S_6), S_3), dim=1))  # [b,256,96,96]
        S_8 = self.c3_u(torch.cat((self.u3(S_7), S_2), dim=1))  # [b,128,192,192]
        S_9 = self.c4_u(torch.cat((self.u4(S_8), S_1), dim=1))  # [b,64,384,384]
        S_out = self.c_out(S_9)  # [b,1,384,384]
        return self.sigmoid(S_out)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 创建模型
    net = DASPP_PCA_Unet().to(device)
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
