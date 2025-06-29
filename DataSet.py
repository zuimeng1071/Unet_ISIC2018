import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


def get_new_data(img, label):
    """

    :param img: Image类型，输入特征图
    :param label: Image类型，单通道
    :return:
    """
    img_arr = np.array(img)
    max_size = max(img.size)
    img_arr = np.pad(img_arr, ((0, max_size - img_arr.shape[0]), (0, max_size - img_arr.shape[1]), (0, 0)),
                     'reflect')

    label_arr = np.array(label)
    label_arr = np.pad(label_arr, ((0, max_size - label_arr.shape[0]), (0, max_size - label_arr.shape[1])),
                       'reflect')
    return img_arr, label_arr


# 数据加载器，后续将加入数据预处理
class Reader(torch.utils.data.Dataset):  # 数据读取
    """
    读取数据
    """

    def __init__(self, images_path, labels_path, transform, nums=None):
        """

        :param images_path: 图片地址
        :param labels_path: 标签地址
        :param transform: 类型转换对象
        """
        super().__init__()
        # 获取数据列表
        self.transform = transform
        datas = []

        new_datas_path = images_path + '_new_datas'
        if not os.path.exists(new_datas_path):
            os.makedirs(new_datas_path)
            os.makedirs(os.path.join(new_datas_path, "img"))
            os.makedirs(os.path.join(new_datas_path, "label"))

        labels_list = os.listdir(labels_path)  # 生成路径
        for i in labels_list:
            if '.png' in i:
                label_path = os.path.join(labels_path, i)
                image_path = os.path.join(images_path, i.split('_')[0] + '_' + i.split('_')[1] + '.jpg')
                datas.append((image_path, label_path))
        if nums is not None:
            datas = datas[:nums]
        print("共{}个数据".format(datas.__len__()))
        self.datas = datas

    def __getitem__(self, item):
        img_path, label_path = self.datas[item]
        # 灰度图
        img = Image.open(img_path)
        label = Image.open(label_path).convert('L')

        img, label = get_new_data(img, label)
        datas = self.transform(image=img, mask=label)
        img = datas['image']
        label = datas['mask'].unsqueeze(dim=0) / 255
        return img.to(torch.float32), label.to(torch.float32)

    def __len__(self):
        return self.datas.__len__()

