import cv2
import numpy as np
import torch


def Image_post_processing(input_datas: torch.Tensor, threshold=0.35, kernel_size=15, iterations=2) -> torch.Tensor:
    """
    数据后处理：先二值化，再进行形态学孔洞填充和膨胀操作，最终输出范围归一化为[0, 1]

    :param input_datas: 输入类型为Tensor，通常为浮点数，范围在[0, 1]之间
    :param threshold: 二值化阀值
    :param kernel_size: 形态学操作的核大小
    :param iterations: 形态学操作的迭代次数
    :return: 处理后的Tensor，类型为float，范围在[0, 1]之间
    """
    # 确保输入为float类型并且在[0, 1]范围内
    input_datas = input_datas.float()
    if input_datas.min() < 0 or input_datas.max() > 1:
        raise ValueError("输入应该在[0, 1]之间")

    processed_datas = []  # 存储每个处理后的数据

    for input_data in input_datas:
        # 将张量转换为NumPy数组
        numpy_data = input_data.squeeze().cpu().numpy()

        # 二值化处理
        binary_data = (numpy_data > threshold).astype(np.uint8) * 255
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 孔洞填充操作（闭操作）
        closed_data = cv2.morphologyEx(binary_data, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        # （开操作）
        opened_data = cv2.morphologyEx(closed_data, cv2.MORPH_OPEN, kernel, iterations=iterations)

        processed_datas.append(opened_data)

    # 将处理后的 NumPy 数组转换回 torch.Tensor
    processed_tensor = torch.from_numpy(np.array(processed_datas)).to(input_datas.device)
    processed_tensor = processed_tensor.reshape(input_datas.shape)
    processed_tensor = processed_tensor.float() / 255.0

    return processed_tensor


if __name__ == "__main__":
    # 测试随机生成的Tensor
    a = torch.rand((1, 1, 512, 512))  # 模拟输入张量
    processed_a = Image_post_processing(a, threshold=0.5, kernel_size=5, iterations=2)
    print(f"Processed shape: {processed_a.shape}")
    print(f"Processed tensor (min={processed_a.min()}, max={processed_a.max()}): {processed_a}")
