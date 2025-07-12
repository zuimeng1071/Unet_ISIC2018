import os
import torch

from models.AtteUnet import AttU_Net
from models.ChannelAtte_Unet import ChannelAtte_Unet
from models.DASPP_Unet import DASPP_Unet
from models.NestedUNet import NestedUNet
from models.R2U import R2U_Net
from models.ResUnet import ResUNet
from models.Unet import Unet


def load_model(model_name: str, image_size=(400, 400), isLoadWeight=False, weight_path=None):
    """
    加载模型函数
    :param model_name: 需要加载的模型 包含 Unet DASPP_Unet ChannelAtte_Unet ResUnet R2U NestedUnet AtteUnet
    :param image_size: 输入到模型中的图片尺寸，部分模型有要求的最小尺寸
    :param isLoadWeight: 是否加载原有的权重
    :param weight_path: 模型权重文件路径
    :return: 加载好的模型
    """
    weight_dir = './weight'
    # 检查图片尺寸是否符合要求
    if image_size[0] % 16 != 0 and image_size[1] % 16 != 0:
        print("图片尺寸不等于16的倍数，这可能会导致模型报错")

    # 根据模型名称初始化模型实例
    if model_name == 'Unet':
        model = Unet()
    elif model_name == 'DASPP_Unet':
        if image_size[0] <= 192 and image_size[1] <= 192:
            print("图片尺寸过小，该模型要求最小尺寸需大于192*192")
            return None
        model = DASPP_Unet()
    elif model_name == 'ChannelAtteUnet':
        model = ChannelAtte_Unet()
    elif model_name == 'R2U':
        model = R2U_Net(3, 1, 1)
    elif model_name == 'NestedUnet':
        model = NestedUNet(1, 3)
    elif model_name == 'AtteUnet':
        model = AttU_Net()
    elif model_name == 'ResUnet':
        model = ResUNet()
    else:
        print(f"未知的模型名称: {model_name}")
        return None

    # 创建权重目录（如果不存在）
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        print('生成目录')

    # 如果模型未成功初始化则返回None
    if model is None:
        return None

    # 如果需要加载权重文件，则尝试从指定路径加载
    if isLoadWeight:
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
            print('加载权重文件成功')
        else:
            print('加载权重文件失败')
            return None

    return model



