import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
from torch.nn import functional as F

from load_model import load_model
from Image_post_processing import Image_post_processing  # 导入新的后处理模块


def preprocess_image(image, image_size):
    """优化后的预处理函数"""

    def get_new_data(img_arr: np.ndarray):
        max_size = max(img_arr.shape[:2])
        return np.pad(img_arr,
                      ((0, max_size - img_arr.shape[0]),
                       (0, max_size - img_arr.shape[1]),
                       (0, 0)),
                      'constant', constant_values=255)

    in_image = np.array(image)
    in_size = in_image.shape[:2]  # 原始尺寸 (H, W)

    # 处理灰度图和填充
    image_arr = get_new_data(in_image)
    if len(image_arr.shape) == 2:
        image_arr = np.stack((image_arr,) * 3, axis=-1)

    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),
        A.MedianBlur(blur_limit=5, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.6),
        A.Normalize(mean=(0.45, 0.5, 0.5), std=(0.5, 0.33, 0.33)),
        ToTensorV2()
    ])

    return transform(image=image_arr)['image'].unsqueeze(0), in_size


def pre(model_name, in_image, image_size=(400, 400), isUseBestWeight=False):
    """整合后处理模块的预测函数"""
    # 加载模型
    model = load_model(model_name, image_size, isLoadWeight=True, isUseBestWeight=isUseBestWeight)
    if model is None:
        return None, None

    # 预处理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_data, in_size = preprocess_image(in_image, image_size)
    image_data = image_data.to(device)
    model = model.to(device).eval()

    # 模型预测
    with torch.no_grad():
        pred = model(image_data)

    # 使用新的后处理模块
    processed = Image_post_processing(
        input_datas=pred,
        threshold=0.3
    )
    # processed = pred

    # 尺寸恢复处理
    max_size = max(in_size)
    processed = F.interpolate(processed, size=(max_size, max_size), mode='nearest')
    processed = processed[:, :, :in_size[0], :in_size[1]]  # 精确裁剪

    # 转换为PIL图像
    result_np = (processed.squeeze().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(result_np), result_np


# 示例使用
if __name__ == "__main__":
    img = Image.open('tools/test.jpg')
    out_img, arr = pre('OrthoProj_UNet', img)
    out_img.save('./测试工具/out.jpg')