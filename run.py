import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, jaccard_score, mean_squared_error, \
    confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from BCE_DiceLoss import BCE_DiceLoss
from Image_post_processing import Image_post_processing
from old_models.DASPP_PCA_GPU_Unet import DASPP_PCA_Unet
from old_models.DASPP_Unet import DASPP_Unet
from old_models.PCA_Unet import PCA_Unet
from models.Unet import Unet
# from Dataset import Reader
from 临时测试文件.Dataset import Reader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ----------------- 评估指标函数 -----------------
def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    y_pred_binary = (y_pred_flat >= 0.5).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_binary).ravel()
    specificity = tn / (tn + fp + 1e-6)
    sensitivity = recall_score(y_true_flat, y_pred_binary)
    precision = precision_score(y_true_flat, y_pred_binary)
    f1 = f1_score(y_true_flat, y_pred_binary)
    jaccard = jaccard_score(y_true_flat, y_pred_binary)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    accuracy = accuracy_score(y_true_flat, y_pred_binary)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "jaccard": jaccard,
        "dice": dice,
        "mse": mse
    }


# ----------------- 绘制损失曲线函数 -----------------
def draw_loss(train_loss, val_loss, save_path):
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='b')
    plt.plot(epochs, val_loss, label='Val Loss', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练和验证损失曲线保存到: {save_path}")
    plt.close()


# ----------------- 选择模型函数 -----------------
def chosModel(model_name, image_size):
    if model_name == 'Unet':
        return Unet()
    elif model_name == 'PCA_Unet':
        return PCA_Unet()
    elif model_name == 'DASPP_Unet':
        if image_size[0] <= 192 or image_size[1] <= 192:
            raise ValueError("图片尺寸过小，该模型要求最小尺寸需大于192x192")
        return DASPP_Unet()
    elif model_name == 'DASPP_PCA_Unet':
        if image_size[0] <= 192 or image_size[1] <= 192:
            raise ValueError("图片尺寸过小，该模型要求最小尺寸需大于192x192")
        return DASPP_PCA_Unet()
    else:
        raise ValueError(f"未知的模型名：{model_name}")


def train_and_val(model_name, images_path, labels_path, batch_size=4, image_size=(400, 400),
                  num_epochs=50, learning_rate=0.001, T_0=10, T_mult=2, eta_min=1e-6, max_norm=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------- 初始化模型和数据 -----------------
    model = chosModel(model_name, image_size).to(device)

    # 数据增强（训练集和验证集分开）
    train_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.ShiftScaleRotate(shift_limit=0.0525, scale_limit=(-0.1, 0.8), rotate_limit=45,
                           interpolation=1, border_mode=4, p=0.45),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

    # 数据集和数据加载器
    train_dataset = Reader(images_path=images_path, labels_path=labels_path,
                           transform=train_transform, mode='train')
    val_dataset = Reader(images_path=images_path, labels_path=labels_path,
                         transform=val_transform, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=4, shuffle=True, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            num_workers=4, shuffle=False, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # ----------------- 优化器和损失函数 -----------------
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    loss_fn = BCE_DiceLoss()

    # ----------------- 训练和验证循环 -----------------
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # ========== 训练阶段 ==========
        model.train()
        epoch_train_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Train'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            # 打印最大梯度和损失
            if torch.isnan(loss):
                print("梯度出现NaN，已跳过当前批次")
                continue
            else:
                print(f"当前批次的损失: {loss.item()}")
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)  # 按样本数加权

        epoch_train_loss /= len(train_loader)  # 平均训练损失
        train_losses.append(epoch_train_loss)

        # ========== 验证阶段 ==========
        model.eval()
        epoch_val_loss = 0.0
        metrics_accumulator = {"accuracy": [], "sensitivity": [], "specificity": [], "precision": [], "f1": [],
                               "jaccard": [], "dice": [], "mse": []}

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images, labels = images.to(device), labels.to(device)
                outputs_raw = model(images)
                outputs = Image_post_processing(outputs_raw, threshold=0.15)
                loss = loss_fn(outputs_raw, labels)

                epoch_val_loss += loss.item()
                metrics = compute_metrics(labels.cpu().numpy(), outputs.cpu().numpy())
                for k, v in metrics.items():
                    metrics_accumulator[k].append(v)

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        # ========== 打印和保存 ==========
        avg_metrics = {k: np.mean(v) for k, v in metrics_accumulator.items()}
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')
        for metric, value in avg_metrics.items():
            print(f'{metric.capitalize()}: {value:.4f}', end=' | ')
        print()

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f'./weight/{model_name}_best.pth')

    # 保存最终模型
    torch.save(model.state_dict(), f'./weight/{model_name}_final.pth')
    torch.cuda.empty_cache()

    return train_losses, val_losses


# ----------------- 主函数 -----------------
if __name__ == "__main__":
    images_path = r"F:\数据集\1-2_Training_Input\ISIC2018_Task1-2_Training_Input"
    labels_path = r"F:\数据集\T1_Training_GroundTruth\ISIC2018_Task1_Training_GroundTruth"

    model_name = "PCA_Unet"

    train_loss, val_loss = train_and_val(model_name, images_path, labels_path, batch_size=1, num_epochs=100,
                                         learning_rate=0.001)

    draw_loss(train_loss, val_loss, save_path=f'{model_name}_loss_comparison.png')
