from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

# 加载并预处理图像
img_fn = "ISIC_0012627.jpg"
img = Image.open(img_fn)


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


# 创建并保存图像
final_image = Image.fromarray(apply_pca_to_image(img))
final_image.save("test_pca.jpg")