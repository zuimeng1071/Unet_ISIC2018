import os.path
import torch
import torchvision
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataSet import Reader
from Image_post_processing import Image_post_processing
from load_model import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_unique_folder_name(base_path, model_name):
    # 构建初始文件夹名称
    folder_name = f'val_{model_name}'
    unique_folder_name = folder_name
    counter = 1

    # 检查文件夹是否已经存在
    while os.path.exists(os.path.join(base_path, unique_folder_name)):
        # 如果文件夹存在，则在文件夹名称后附加递增的数字
        unique_folder_name = f'{folder_name}_{counter}'
        counter += 1

    # 返回唯一的文件夹路径
    return os.path.join(base_path, unique_folder_name)


def compute_metrics(y_true, y_pred):
    """
    计算各种评估指标
    :param y_true: 真实标签
    :param y_pred: 预测结果
    :return: 各种评估指标字典
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    TP = np.sum((y_pred_flat == 1) & (y_true_flat == 1))  # TP: 真正例
    TN = np.sum((y_pred_flat == 0) & (y_true_flat == 0))  # TN: 真负例
    FP = np.sum((y_pred_flat == 1) & (y_true_flat == 0))  # FP: 假正例
    FN = np.sum((y_pred_flat == 0) & (y_true_flat == 1))  # FN: 真反例

    # 计算评估指标
    # 准确率=(TP+TN)/(TP+TN+FP+FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0
    # 敏感度=(TP)/(TP+FN)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    # 特异度=(TN)/(TN+FP)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
    # 精准率=(TP)/(TP+FP)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    # F1=(2*TP)/(2*TP+FP+FN)
    f1 = f1_score(y_true_flat, y_pred_flat)
    # Jaccard系数=(TP)/(TP+FP+FN)
    iou = jaccard_score(y_true_flat, y_pred_flat)
    # Dice系数=(2*TP)/(2*TP+FP+FN)
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0.0
    # MSE=(y_true-y_pred)^2
    mse = np.mean((y_true - y_pred) ** 2)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "jaccard": iou,
        "dice": dice,
        "mse": mse
    }


def create_data_loader(val_images_path, val_label_path, image_size, nums=None):
    """
    创建验证集的数据加载器
    :param val_images_path: 测试集图片地址
    :param val_label_path: 测试集图片标签地址
    :param image_size: 图片尺寸
    :param nums: 验证集个数
    :return: 数据加载器
    """
    transform = A.Compose([
        # 预处理模块
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),  # 对比度增强
        A.MedianBlur(blur_limit=5, p=0.5),  # 毛发噪声抑制
        A.RandomGamma(gamma_limit=(80, 120), p=0.6),  # 伽马校正
        # 空间几何变换
        A.Resize(image_size[0], image_size[1]),  # 调整图像大小到目标尺寸
        # 归一化与格式转换
        A.Normalize(mean=(0.45, 0.5, 0.5), std=(0.5, 0.33, 0.33)),  # 图像归一化
        ToTensorV2()  # 将图像转换为 PyTorch 张量
    ], additional_targets={'mask': 'mask'})  # 定义额外的目标（如掩码）的变换方式

    val_dataset = Reader(images_path=val_images_path, labels_path=val_label_path, transform=transform, nums=nums)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=0, shuffle=False)
    return val_dataloader


def validate_model(model, val_dataloader, save_images_path):
    """
    在验证集上进行模型验证
    :param model: 已经加载好权重的模型实例
    :param val_dataloader: 验证集数据加载器
    :param save_images_path: 保存预测结果图片的路径
    :return: 平均评估指标
    """
    metrics_accumulator = {"accuracy": [], "sensitivity": [], "specificity": [], "precision": [], "f1": [],
                           "jaccard": [], "dice": [], "mse": []}

    i = 0
    for images, labels in tqdm(val_dataloader, desc='Validating'):
        i += 1
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        out = model(images)
        out = Image_post_processing(out, threshold=0.3)

        # 转换到CPU并转为numpy数组
        out_np = out.cpu().numpy().squeeze()
        labels_np = labels.cpu().numpy().squeeze()

        # 处理batch维度
        batch_size = 1 if len(out_np.shape) == 2 else out_np.shape[0]
        if batch_size == 1:
            out_np = np.expand_dims(out_np, axis=0)
            labels_np = np.expand_dims(labels_np, axis=0)

        for j in range(batch_size):
            # 获取单个样本数据
            pred = out_np[j]
            true = labels_np[j]

            # 获取结果
            binary_pred = (pred > 0.5).astype(np.uint8)
            binary_true = (true > 0.5).astype(np.uint8)

            # 计算各项指标
            metrics = compute_metrics(binary_true, binary_pred)
            for k, v in metrics.items():
                metrics_accumulator[k].append(v)
            # 保存图片
            img = torch.stack((images[0], torch.cat((labels[0], labels[0], labels[0]), dim=0),
                               torch.cat((out[0], out[0], out[0]), dim=0)), dim=0)
            torchvision.utils.save_image(img,
                                         os.path.join(save_images_path,
                                                      f'{i}' + f'_val.jpg'))

    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics_accumulator.items()}
    return avg_metrics


def val(model_name: str, val_images_path: str, val_label_path: str,
        image_size=(1, 400, 400), isUseBestWeight=False,
        nums=None):
    """
    进行模型验证的主要函数
    :param nums: 验证集个数
    :param model_name: 选择的模型 包含 'Unet' 和 'DASPP_PCA_Unet'
    :param val_images_path: 测试集图片地址
    :param val_label_path: 测试集图片标签地址
    :param image_size: 图片尺寸，部分模型有要求的最小尺寸
    :param isUseBestWeight: 是否使用训练时的最佳权重
    :return:
    """
    save_images_path = get_unique_folder_name('./save_img', model_name)
    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path)
        print('生成目录')

    model = load_model(model_name, image_size, isLoadWeight=True, isUseBestWeight=isUseBestWeight)
    if model is None:
        return

    model = model.to(device)
    model.eval()

    val_dataloader = create_data_loader(val_images_path, val_label_path, image_size, nums)
    avg_metrics = validate_model(model, val_dataloader, save_images_path)

    print(f"\nValidation Results for {model_name}:")
    for metric, value in avg_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}", end=' | ')
    print()


if __name__ == "__main__":
    images_val_path = r"E:\数据集\ISIC2018_Task1-2_Validation_Input"
    # images_val_path = r"E:\数据集\Val_Input_PCA"
    label_val_path = r"E:\数据集\ISIC2018_Task1_Validation_GroundTruth"
    with torch.no_grad():
        val("Unet",
            images_val_path, label_val_path,
            image_size=(400, 400), isUseBestWeight=False)
