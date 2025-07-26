import gc
import os.path
from datetime import datetime
import albumentations as A
import cv2
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入自定义的损失函数和模型类
from BCE_DiceLoss import BCE_DiceLoss
from DataSet import Reader
from load_model import load_model

# 设置环境变量以避免KMP_DUPLICATE_LIB_OK错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
base_weight_path = "./weight"

def draw(x_data, y_data_dict, save_path=None):
    """
    绘制并保存多个指标的变化曲线
    :param x_data: X轴数据，通常是epoch数
    :param y_data_dict: 字典，键是指标名称，值是对应的Y轴数据列表
    :param save_path: 如果提供了保存路径，则将图形保存到该路径；否则显示图形
    """
    plt.figure(figsize=(10, 6))
    for label, y_data in y_data_dict.items():
        plt.plot(x_data, y_data, label=label)

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"折线图保存到指定路径: {save_path}")
        plt.close()  # 关闭图形以释放资源
    else:
        plt.show()


def get_unique_folder_name(base_path):
    # 构建初始文件夹名称
    folder_name = f'train'
    unique_folder_name = folder_name
    counter = 1

    # 检查文件夹是否已经存在
    while os.path.exists(os.path.join(base_path, unique_folder_name)):
        # 如果文件夹存在，则在文件夹名称后附加递增的数字
        unique_folder_name = f'{folder_name}_{counter}'
        counter += 1
    # 创建目录
    os.makedirs(os.path.join(base_path, unique_folder_name), exist_ok=True)

    # 返回唯一的文件夹路径
    return os.path.join(base_path, unique_folder_name)


def get_transform(image_size):
    """
    获取图像预处理变换管道

    :param image_size: 目标图像尺寸 (height, width)
    :return: Albumentations 变换管道
    """
    return A.Compose([
        # 预处理模块
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),  # 对比度增强
        A.MedianBlur(blur_limit=5, p=1),  # 毛发噪声抑制
        A.RandomGamma(gamma_limit=(80, 120), p=1),  # 伽马校正

        # 空间几何变换
        A.Resize(image_size[0], image_size[1]),  # 调整图像大小到目标尺寸
        # 归一化与格式转换
        A.Normalize(mean=(0.45, 0.5, 0.5), std=(0.5, 0.33, 0.33)),  # 图像归一化
        ToTensorV2()  # 将图像转换为 PyTorch 张量
    ], additional_targets={'mask': 'mask'})  # 定义额外的目标（如掩码）的变换方式


def save_temp_weight(model):
    """
    保存临时权重文件
    """
    # 获取当前时间，并格式化为合法的文件名格式
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 构造权重文件路径
    weight_path = os.path.join('./weight', f'temp_{type(model).__name__}_{current_time}_weight.pth')
    torch.save(model.state_dict(), weight_path)
    print("保存临时权重文件:", weight_path)


def initialize_optimizer_and_scheduler(model, lr):
    """
    初始化优化器和学习率调度器（单调递减）
    :param model: 模型实例
    :param lr: 学习率
    :return: 优化器和学习率调度器
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    step_size = 5  # 每n个epoch下降一次学习率
    gamma = 0.5  # 下降系数
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return optimizer, scheduler


def create_data_loader(images_path, labels_path, transform, batch_size):
    """
    创建数据加载器
    :param images_path: 图片地址
    :param labels_path: 图片标签地址
    :param transform: 数据增强变换序列
    :param batch_size: 批次大小
    :return: 数据加载器
    """
    dataset = Reader(images_path=images_path,
                     labels_path=labels_path,
                     transform=transform)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=3,
                            shuffle=True)
    return dataloader


def calculate_metrics(output, labels):
    """
    计算准确率、精确率、召回率和F1分数
    :param output: 模型输出
    :param labels: 标签
    :return: 准确率、精确率、召回率和F1分数
    """
    predicted = (output > 0.5).float()
    correct = (predicted == labels).sum().item()
    total = labels.numel()

    accuracy = correct / total

    true_positives = ((predicted == 1) & (labels == 1)).sum().item()
    false_positives = ((predicted == 1) & (labels == 0)).sum().item()
    false_negatives = ((predicted == 0) & (labels == 1)).sum().item()

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    return accuracy, precision, recall, f1_score


def compute_average_gradient(model):
    """
    计算模型参数的平均梯度
    :param model: 模型实例
    :return: 平均梯度
    """
    total_norm = 0.0
    count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm()
            total_norm += param_norm.item()
            count += 1
    if count == 0:
        return 0.0
    return total_norm / count


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, epoch, step_data, metrics_data, grad_data,
                lr_data, save_images_dir):
    """
    单个训练轮次
    :param model: 模型实例
    :param dataloader: 数据加载器
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param device: 设备（CPU或GPU）
    :param epoch: 当前轮次编号
    :param step_data: 记录步骤的数据列表
    :param metrics_data: 记录各种指标的数据字典
    :param grad_data: 记录平均梯度的数据列表
    :param lr_data: 记录学习率的数据列表
    :param save_images_dir: 保存样本图片的目录
    :return: 平均损失
    """
    running_loss = 0
    avg_loss = 0
    sum_count = 0
    steps_per_epoch = len(dataloader)

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0

    model.train()
    epoch_tqdm = tqdm(dataloader, ncols=150, desc=f'epoch:{epoch}')

    for j, datas in enumerate(epoch_tqdm):
        try:
            # 数据处理
            images, labels = datas
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            # 模型学习
            optimizer.zero_grad()  # 清零梯度
            ret_loss = loss_fn(output, labels)  # 计算损失
            ret_loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            ret_loss_item = ret_loss.item()
            running_loss += ret_loss_item
            avg_loss += ret_loss_item
            sum_count += 1

            # 计算指标
            accuracy, precision, recall, f1_score = calculate_metrics(output, labels)
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score

            # 显示损失和其他指标
            epoch_tqdm.set_description(
                f"loss:{ret_loss_item:0.6e}, "
                f"acc:{accuracy:.4f}, "
                f"prec:{precision:.4f}, "
                f"rec:{recall:.4f}, "
                f"f1:{f1_score:.4f}, "
                f"lr:{optimizer.param_groups[0]['lr']:.6e}, "
                f"epoch:{epoch + 1}"
            )
            epoch_tqdm.update(1)

            # 记录学习率
            lr_data.append(optimizer.param_groups[0]['lr'])

            # 定期保存样例
            if j % 200 == 0:
                img = torch.stack((images[0], torch.cat((labels[0], labels[0], labels[0]), dim=0),
                                   torch.cat((output[0], output[0], output[0]), dim=0)), dim=0)
                torchvision.utils.save_image(img,
                                             os.path.join(save_images_dir,
                                                          f'{j}' +
                                                          f'_train.jpg'))

                running_loss = 0
                sum_count = 0
        except KeyboardInterrupt as e:
            save_temp_weight(model)
            print(f"训练过程中发生KeyboardInterrupt错误: {e}")
        except RuntimeError as e:
            save_temp_weight(model)
            print(f"训练过程中发生RuntimeError错误: {e}")
        except ValueError as e:
            save_temp_weight(model)
            print(f"训练过程中发生ValueError错误: {e}")
        except Exception as e:
            save_temp_weight(model)
            print(f"训练过程中发生未知错误: {e}")
    else:
        # 保存权重
        # 调整学习率
        scheduler.step()
        print(f"保存临时权重, epoch:{epoch + 1}")
        weight_path = os.path.join('./weight', f'temp_{type(model).__name__}_weight.pth')
        torch.save(model.state_dict(), weight_path)

    # 记录数据
    metrics_data['loss'].append(avg_loss / steps_per_epoch)
    metrics_data['accuracy'].append(total_accuracy / steps_per_epoch)
    metrics_data['precision'].append(total_precision / steps_per_epoch)
    metrics_data['recall'].append(total_recall / steps_per_epoch)
    metrics_data['f1_score'].append(total_f1_score / steps_per_epoch)
    step_data.append(epoch + 1)

    # 计算并记录平均梯度
    avg_grad = compute_average_gradient(model)
    grad_data.append(avg_grad)

    return avg_loss / steps_per_epoch


def Train(model, images_path: str, labels_path: str,
          isSaveWeight=False,
          epoch_count=50,
          lr=0.01,
          batch_size=1,
          image_size=(400, 400)):
    """
    模型训练函数，模型为Unet及其变形
    :param model: 已经初始化并可能已经加载权重的模型实例
    :param images_path: 图片地址
    :param labels_path: 图片标签地址
    :param isSaveWeight: 是否保存训练后的权重
    :param epoch_count: 训练次数，1个epoch为遍历整个数据集的完整周期
    :param lr: 学习率
    :param batch_size: 批次，如果显存不足可以适当降低
    :param image_size: 输入到模型中的图片尺寸，部分模型有要求的最小尺寸
    :return:
    """

    # 构造保存训练样本图片的目录
    save_images_dir = '_'.join(['./save_img/train', type(model).__name__])
    if not os.path.exists(save_images_dir):
        os.makedirs(save_images_dir)
        print(f'生成目录:{save_images_dir}')
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"./run/model_run_{current_time}"

    # 创建模型运行结果保存目录
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        print(f'生成目录:{run_dir}')

    # 如果模型加载失败则退出训练
    if model is None:
        print(f'{type(model).__name__}: 模型加载失败')
        return

    # 将模型移动到GPU或CPU设备上
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义数据增强策略
    transform = get_transform(image_size)

    # 初始化数据集和数据加载器
    dataloader = create_data_loader(images_path, labels_path, transform, batch_size)

    # 定义损失函数和优化器
    loss_fn = BCE_DiceLoss()  # 混合损失函数 BCE_DiceLoss
    optimizer, scheduler = initialize_optimizer_and_scheduler(model, lr)

    # 初始化记录损失的数据列表
    metrics_data = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    step_data = []
    grad_data = []
    lr_data = []
    min_loss = float('inf')
    train_count = 0

    try:
        # 获取唯一的权重保存目录
        weight_dir = get_unique_folder_name(base_weight_path)

        # 构造完整的权重文件路径（不立即写入）
        best_weight_path = os.path.join(weight_dir, f"best_{type(model).__name__}_weight.pth")
        final_weight_path = os.path.join(weight_dir, f"end_{type(model).__name__}_weight.pth")

        min_loss = float('inf')

        for i in range(epoch_count):
            avg_loss = train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, i,
                                   step_data, metrics_data, grad_data, lr_data, save_images_dir)
            # 保存最佳权重和最终权重
            if isSaveWeight:
                # 保存最终权重（每个 epoch 都覆盖）
                torch.save(model.state_dict(), final_weight_path)

                # 如果当前 loss 更小，则保存为最佳权重
                if avg_loss < min_loss:
                    min_loss = avg_loss
                    torch.save(model.state_dict(), best_weight_path)

            # 内存优化
            gc.collect()
            torch.cuda.empty_cache()
            train_count += 1
    except KeyboardInterrupt:
        save_temp_weight(model)
        print("用户中断")
    except Exception as e:
        save_temp_weight(model)
        print(f"训练过程中发生未知错误: {e}")
    # 绘制损失曲线
    draw(
        x_data=step_data,
        y_data_dict={'loss': metrics_data['loss']},
        save_path=f'./{run_dir}/{type(model).__name__}_loss.png'
    )

    # 绘制其他指标（准确率、精确率、召回率、F1）
    other_metrics = {k: v for k, v in metrics_data.items() if k != 'loss'}
    draw(
        x_data=step_data,
        y_data_dict=other_metrics,
        save_path=f'./{run_dir}/{type(model).__name__}_metrics.png'
    )

    # 绘制梯度曲线
    draw(
        x_data=step_data,
        y_data_dict={'gradient': grad_data},
        save_path=f'./{run_dir}/{type(model).__name__}_gradient.png'
    )

    # 绘制学习率曲线（使用batch索引作为x轴）
    draw(
        x_data=range(len(lr_data)),
        y_data_dict={'learning_rate': lr_data},
        save_path=f'./{run_dir}/{type(model).__name__}_learning_rate.png'
    )


if __name__ == "__main__":
    images_path = r"E:\数据集\Training_Input_PCA"
    # images_path = r"E:\数据集\ISIC2018_Task1-2_Training_Input"
    label_path = r"E:\数据集\ISIC2018_Task1_Training_GroundTruth"
    # images_path = r"E:\数据集\ISIC2018_Task1-2_Validation_Input"
    # label_path = r"E:\数据集\ISIC2018_Task1_Validation_GroundTruth"
    # 加载预训练模型
    model = load_model('DASPP_Unet', image_size=(400, 400), isLoadWeight=False)
    if model is not None:
        # 开始训练
        Train(model, images_path, label_path, batch_size=1, epoch_count=50, lr=0.001, isSaveWeight=True)