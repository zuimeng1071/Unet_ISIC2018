import os
import torch

from Unet_ISIC2018.models.DASPP_ChannelAtte_Unet import DASPP_ChannelAtte_Unet
from models.ChannelAtte_Unet import ChannelAtte_Unet
from models.DASPP_OProj_ChannelAtte_UNet import DASPP_OProj_ChannelAtte_UNet
from models.DASPP_OrthoProj_UNet import DASPP_OrthoProj_UNet
from models.DASPP_Unet import DASPP_Unet
from models.OrthoProj_UNet import OrthoProj_UNet
from models.PCA_Unet import PCA_Unet
from models.Unet import Unet


def load_model(model_name: str, image_size=(400, 400), isLoadWeight=False, weight_path=None):
    """
    加载模型函数
    :param model_name: 需要加载的模型 包含 'Unet', 'DASPP_Unet', 'PCA_Unet', 'DASPP_PCA_Unet',
                       'ChannelAtteUnet', 'DASPP_ChannelAtte_Unet', 'DASPP_OrthoProj_UNet', 'OrthoProj_UNet'
    :param image_size: 输入到模型中的图片尺寸，部分模型有要求的最小尺寸
    :param isLoadWeight: 是否加载原有的权重
    :param isUseBestWeight: 是否加载最佳的权重
    :return: 加载好的模型
    """
    weight_dir = './weight'
    # 检查图片尺寸是否符合要求
    if image_size[0] % 16 != 0 and image_size[1] % 16 != 0:
        print("图片尺寸不等于16的倍数，这可能会导致模型报错")

    # 根据模型名称初始化模型实例
    if model_name == 'Unet':
        model = Unet()
    elif model_name == 'PCA_Unet':
        model = PCA_Unet()
    elif model_name == 'DASPP_Unet':
        if image_size[0] <= 192 and image_size[1] <= 192:
            print("图片尺寸过小，该模型要求最小尺寸需大于192*192")
            return None
        model = DASPP_Unet()
    elif model_name == 'ChannelAtte_Unet':#ChannelAtteUnet
        model = ChannelAtte_Unet()
    elif model_name == 'DASPP_ChannelAtte_Unet':
        if image_size[0] <= 192 and image_size[1] <= 192:
            print("图片尺寸过小，该模型要求最小尺寸需大于192*192")
            return None
        model = DASPP_ChannelAtte_Unet()
    elif model_name == 'DASPP_OrthoProj_UNet':
        if image_size[0] <= 192 and image_size[1] <= 192:
            print("图片尺寸过小，该模型要求最小尺寸需大于192*192")
            return None
        model = DASPP_OrthoProj_UNet()
    elif model_name == 'OrthoProj_UNet':
        model = OrthoProj_UNet()
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



