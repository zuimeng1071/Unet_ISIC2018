import os

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
import cv2
from torch.nn import functional as F

# 假设这些是从其他模块导入的函数
from load_model import load_model
from tools.HotImage_post_processing import HotImage_post_processing


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


def create_output_images(original_image, processed_image):
    """创建三个版本的输出图像：左右拼接、上下拼接、单独热力图"""
    original_image = Image.fromarray(np.uint8(original_image))
    processed_image = Image.fromarray(processed_image)

    # 左右拼接
    side_by_side = Image.new('RGB', (
        original_image.width + processed_image.width, max(original_image.height, processed_image.height)))
    side_by_side.paste(original_image, (0, 0))
    side_by_side.paste(processed_image, (original_image.width, 0))

    # 上下拼接
    top_bottom = Image.new('RGB', (
        max(original_image.width, processed_image.width), original_image.height + processed_image.height))
    top_bottom.paste(original_image, (0, 0))
    top_bottom.paste(processed_image, (0, original_image.height))

    return side_by_side, top_bottom, processed_image


def pre(model, in_image, image_size=(400, 400), isUseBestWeight=False):
    """整合后处理模块的预测函数"""
    if model is None:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_data, in_size = preprocess_image(in_image, image_size)
    image_data = image_data.to(device)
    model = model.to(device).eval()

    with torch.no_grad():
        pred = model(image_data)

    processed = HotImage_post_processing(input_datas=pred)

    max_size = max(in_size)
    processed = F.interpolate(processed, size=(max_size, max_size), mode='nearest')
    processed = processed[:, :, :in_size[0], :in_size[1]]

    result_np = (processed.squeeze().cpu().numpy() * 255).astype(np.uint8)
    result_np = result_np.transpose(1, 2, 0)

    side_by_side, top_bottom, heat_map_only = create_output_images(np.array(in_image), result_np)

    return side_by_side, top_bottom, heat_map_only


def main(input_dir, output_dir):
    # 读取目录下所有文件
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
    model = load_model("DASPP_ChannelAtte_UNet",
                       image_size=(400, 400),
                       isLoadWeight=True,
                       weight_path="weight/end_DASPP_ChannelAtte_Unet_weight.pth")
    for img_path in image_files:
        img = Image.open(img_path)
        side_by_side, top_bottom, heat_map_only = pre(model, img)
        side_by_side.save(os.path.join(output_dir, os.path.basename(img_path)))
        top_bottom.save(os.path.join(output_dir, os.path.basename(img_path)))
        heat_map_only.save(os.path.join(output_dir, os.path.basename(img_path)))


if __name__ == "__main__":
    input_dir = "images"
    output_dir = "hot_images"
    main(input_dir, output_dir)
