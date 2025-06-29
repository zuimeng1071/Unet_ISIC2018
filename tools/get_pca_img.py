import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

# input_dir = r"E:\数据集\ISIC2018_Task1-2_Validation_Input"  # 输入图像目录
# output_dir = r"E:\数据集\Val_Input_PCA"  # 输出图像目录
input_dir = r"E:\数据集\ISIC2018_Task1-2_Training_Input"  # 输入图像目录
output_dir = r"E:\数据集\Training_Input_PCA"  # 输出图像目录

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def apply_pca_to_image(image, n_components=64):
    # 确保图像是numpy数组格式
    image = np.array(image)
    # 将三个(R,G,B)通道简单拼接在一起
    im1 = np.hstack((image[:, :, 0], image[:, :, 1], image[:, :, 2]))

    # 对每个通道进行归一化处理，使每列的均值为0，标准差为1
    means = np.mean(im1, axis=0)
    sds = np.std(im1, axis=0)

    # 防止除以0，给标准差加一个小的常数
    epsilon = 1e-8
    sds[sds == 0] = epsilon

    im2 = (im1 - means) / sds

    # 使用PCA进行降维与重构
    pca = PCA(n_components=n_components)

    # 计算原维度和降维后的维度
    C = pca.fit_transform(im2)  # 进行PCA变换

    # 重构数据
    im3 = pca.inverse_transform(C)

    # 反归一化
    im3 = np.clip(im3 * sds + means, 0, 255)
    im3 = im3.astype('uint8')

    # 重新分割成三个(R,G,B)通道
    im3_channels = np.hsplit(im3, 3)
    im4 = np.zeros_like(image)
    for i in range(3):
        im4[:, :, i] = im3_channels[i]

    return im4.astype('uint8')


# 获取所有待处理的图片路径
image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

for img_path in tqdm(image_files, desc='Processing images'):
    try:
        with Image.open(img_path).convert('RGB') as img:

            # 应用PCA
            img_processed = apply_pca_to_image(img)

            # 保存处理后的图像
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            Image.fromarray(img_processed).save(output_path)
    except Exception as e:
        print(f"处理文件 {img_path} 时出错：{e}")

print("所有图像处理完毕并保存到", output_dir)
