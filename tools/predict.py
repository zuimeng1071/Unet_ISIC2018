import os

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


def pre(model, in_image, image_size=(400, 400), isUseBestWeight=False):
    """整合后处理模块的预测函数"""
    # 加载模型

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


def predict(input_dir, output_dir, model_name, weight_path=None):
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = load_model(model_name, isLoadWeight=True, weight_path=weight_path)
    # 模型预测
    for img_path in image_files:
        img = Image.open(img_path)
        out_img, arr = pre(model, img)
        out_img.save(os.path.join(output_dir, os.path.basename(img_path)))


# 示例使用
if __name__ == "__main__":
    img_path = "test_images/no_pca"
    pca_img_path = "test_images/pca"
    predict(img_path, "pre_out/Unet", "Unet",
            weight_path="weight/end_Unet_weight.pth")
    predict(pca_img_path, "pre_out/pca_Unet", "Unet",
            weight_path="weight/end_pca_Unet_weight.pth")
    predict(img_path, "pre_out/DASPP_Unet", "DASPP_Unet",
            weight_path="weight/end_DASPP_Unet_weight.pth")
    predict(img_path, "pre_out/ChannelAtte_Unet", "ChannelAtteUnet",
            weight_path="weight/end_ChannelAtte_Unet_weight.pth")
    predict(img_path, "pre_out/DASPP_ChannelAtte_Unet", "DASPP_ChannelAtte_UNet",
            weight_path="weight/end_DASPP_ChannelAtte_Unet_weight.pth")
    predict(pca_img_path, "pre_out/pca_DASPP_ChannelAtte_Unet", "DASPP_ChannelAtte_UNet",
            weight_path="weight/end_pca_DASPP_ChannelAtte_Unet_weight.pth")
    predict(img_path, "pre_out/R2U", "R2U",
            weight_path="weight/end_R2U_Net_weight.pth")
    predict(img_path, "pre_out/ResUNet", "ResUnet",
            weight_path="weight/end_ResUNet_weight.pth")
    predict(img_path, "pre_out/NestedUNet", "NestedUnet",
            weight_path="weight/end_NestedUNet_weight.pth")
    predict(img_path, "pre_out/AttUNet", "AtteUnet",
            weight_path="weight/end_AttU_Net_weight.pth")
