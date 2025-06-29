import torch.nn as nn

"""
DiceLoss 和 BCE 相加
"""


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # 将预测结果和标签展平
        outputs = outputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # 计算 Dice 系数
        intersection = (outputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)

        # 返回 Dice Loss
        return 1 - dice_coeff


# 混合损失 BCE + Dice Loss
class BCE_DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, dice_weight=0.33):
        """
        BCE + Dice 混合损失函数
        :param smooth: 用于 DiceLoss 的平滑参数
        :param dice_weight: DiceLoss 的权重比例（控制其对总损失的影响）
        """
        super(BCE_DiceLoss, self).__init__()
        self.bce = nn.BCELoss()  # 二值交叉熵损失
        self.dice = DiceLoss(smooth)  # Dice 损失
        self.dice_weight = dice_weight  # DiceLoss 的权重

    def forward(self, outputs, targets):
        """
        计算 BCE 和 Dice 的混合损失
        :param outputs: 模型的输出 (经过 Sigmoid 激活后)
        :param targets: 目标标签
        :return: 总损失（BCE 损失 + Dice 损失）
        """
        bce_loss = self.bce(outputs, targets)  # 计算 BCE 损失
        dice_loss = self.dice(outputs, targets)  # 计算 Dice 损失
        total_loss = bce_loss + self.dice_weight * dice_loss  # 将 Dice 损失按权重加入
        return total_loss
