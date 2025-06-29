import os
from PIL import Image
import numpy as np


def calculate_stats(image_path):
    # 打开图像并转换为RGB模式（三通道）
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img) / 255.0  # 归一化到 [0, 1]

    # 计算每个通道的均值和标准差
    means = np.mean(img_array, axis=(0, 1))
    std_devs = np.std(img_array, axis=(0, 1))

    return means, std_devs


def process_directory(directory):
    total_means = np.zeros(3)
    total_std_devs = np.zeros(3)
    image_count = 0

    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            means, std_devs = calculate_stats(image_path)

            total_means += means
            total_std_devs += std_devs
            image_count += 1

    avg_means = total_means / image_count
    avg_std_devs = total_std_devs / image_count

    return avg_means, avg_std_devs


images_val_path = r"F:\数据集\1-2_Validation_Input\ISIC2018_Task1-2_Validation_Input"
avg_means, avg_std_devs = process_directory(images_val_path)

print(f"平均均值 (R, G, B): {avg_means}")
print(f"平均标准差 (R, G, B): {avg_std_devs}")
