import torch
import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import onnx
from typing import Optional

from matplotlib import pyplot as plt

from Image_post_processing import Image_post_processing
from load_model import load_model

from torch.onnx import register_custom_op_symbolic


def custom_mT(g, self):
    return g.op("Transpose", self, perm_i=[1, 0])


def torch_to_onnx(model_name: str,
                  image_size: tuple = (400, 400),
                  onnx_path: str = './OrthoProj_UNet.onnx',
                  opset_version: int = 11) -> Optional[str]:
    """
    优化后的ONNX导出函数，支持动态尺寸
    """
    try:
        # 加载模型并设为评估模式
        model = load_model(model_name, image_size, isLoadWeight=False, isUseBestWeight=False)
        model.eval()

        # 创建动态输入示例（H和W可动态变化）
        dummy_input = torch.randn(1, 3, *image_size)

        # 导出配置
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {2: 'height', 3: 'width'},
                'output': {2: 'height_out', 3: 'width_out'}
            }
        )

        # 模型验证
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX导出成功: {onnx_path}")
        return onnx_path

    except Exception as e:
        print(f"导出失败: {str(e)}")
        return None


def inference_onnx(onnx_path: str,
                   image_path: str,
                   image_size: tuple = (400, 400)) -> np.ndarray:
    """
    带后处理的ONNX推理函数
    """
    # 预处理（保持与训练一致）
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.5, 0.5], std=[0.5, 0.33, 0.33])
    ])

    with Image.open(image_path) as img:
        original_size = img.size
        input_tensor = transform(img.convert('RGB')).unsqueeze(0).numpy()

    # ONNX推理
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # 后处理集成
    processed = Image_post_processing(
        torch.from_numpy(ort_output),
        threshold=0.3
    ).squeeze().numpy()

    # 恢复原始尺寸
    resized_mask = Image.fromarray(processed).resize(original_size, Image.NEAREST)
    return np.array(resized_mask)


def visualize_result(image_path: str, mask_array: np.ndarray):
    """
    可视化工具函数
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_array, cmap='gray')
    plt.title('Segmentation Result')

    plt.show()


if __name__ == "__main__":
    register_custom_op_symbolic("aten::mT", custom_mT, 17)
    # 步骤1：导出ONNX模型
    onnx_path = torch_to_onnx(
        model_name='OrthoProj_UNet',
        image_size=(400, 400),
        onnx_path='./OrthoProj_UNet_dynamic.onnx'
    )

    if onnx_path:
        # 步骤2：测试推理
        test_image = './test.jpg'
        result = inference_onnx(onnx_path, test_image)

        # 步骤3：可视化结果
        visualize_result(test_image, result)
