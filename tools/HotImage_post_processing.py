import cv2
import numpy as np
import torch


def HotImage_post_processing(input_datas: torch.Tensor) -> torch.Tensor:
    """
    数据后处理：生成彩色热力图
    :param input_datas: 输入类型为Tensor，浮点数，范围在[0, 1]之间
    :return: 处理后的Tensor，形状为 [C, H, W]
    """
    # 假设输入是 [B, C, H, W]，这里取第一个样本
    input_data = input_datas[0].cpu().numpy()  # 转为 numpy

    # 如果是单通道 [1, H, W]，去掉通道维度
    if input_data.shape[0] == 1:
        input_data = input_data[0]  # 变为 [H, W]
    else:
        raise ValueError("Expected single-channel input tensor.")

    # 归一化到 [0, 255] 并转为 uint8
    img_uint8 = (input_data * 255).astype(np.uint8)

    # 应用 OpenCV 的伪彩色映射（热力图）
    heatmap = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 转为 Tensor 并调整维度 [H, W, C] -> [C, H, W]
    heatmap_tensor = torch.from_numpy(heatmap).permute(2, 0, 1).float() / 255.0

    # 增加 batch 维度 [C, H, W] -> [1, C, H, W]
    heatmap_tensor = heatmap_tensor.unsqueeze(0)

    return heatmap_tensor


if __name__ == "__main__":
    # 测试随机生成的Tensor
    a = torch.rand((1, 1, 512, 512))  # 模拟输入张量
    processed_a = HotImage_post_processing(a)
    print(f"Processed shape: {processed_a.shape}")
    print(f"Processed tensor (min={processed_a.min()}, max={processed_a.max()}): {processed_a}")
