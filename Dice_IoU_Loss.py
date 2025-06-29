import torch.nn as nn

"""
DiceLoss 和 IoULoss 相加
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


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # 将预测结果和标签展平
        outputs = outputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # 计算交集和并集
        intersection = (outputs * targets).sum()
        union = (outputs.sum() + targets.sum()) - intersection

        # 计算 IoU
        iou = (intersection + self.smooth) / (union + self.smooth)

        # 返回 IoU Loss
        return 1 - iou


# 定义混合损失 Dice + IoU Loss
class Dice_IoULoss(nn.Module):
    def __init__(self, smooth=1e-6, iou_weight=0.33):
        """
        Dice + IoU 混合损失函数
        :param smooth: 用于 DiceLoss 和 IoULoss 的平滑参数
        :param iou_weight: IoULoss 的权重比例（控制其对总损失的影响）
        """
        super(Dice_IoULoss, self).__init__()
        self.dice = DiceLoss(smooth)  # Dice 损失
        self.iou = IoULoss(smooth)  # IoU 损失
        self.iou_weight = iou_weight  # IoULoss 的权重

    def forward(self, outputs, targets):
        """
        计算 Dice 和 IoU 的混合损失
        :param outputs: 模型的输出 (经过 Sigmoid 激活后)
        :param targets: 目标标签
        :return: 总损失（Dice 损失 + IoU 损失）
        """
        dice_loss = self.dice(outputs, targets)  # 计算 Dice 损失
        iou_loss = self.iou(outputs, targets)  # 计算 IoU 损失
        total_loss = dice_loss + self.iou_weight * iou_loss  # 将 IoU 损失按权重加入
        return total_loss
