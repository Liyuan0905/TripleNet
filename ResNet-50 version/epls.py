import os
import torch
from torchvision import transforms
from PIL import Image

# 设置文件夹路径
folder1 = './res_contour'
folder2 = './res_body'
folder3 = './res_pred'

output_folder = './epls'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 定义图像的转换：将图片转换为张量
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# 获取两个文件夹中所有图片的文件名交集
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))
files3 = set(os.listdir(folder3))

common_files = files1.intersection(files2)

for filename in common_files:
    # 加载两个文件夹中的图片
    image1_path = os.path.join(folder1, filename)
    image2_path = os.path.join(folder2, filename)
    image3_path = os.path.join(folder3, filename)
    
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')
    image3 = Image.open(image3_path).convert('RGB')
    
    # 转换为张量
    tensor1 = to_tensor(image1)
    tensor2 = to_tensor(image2)
    tensor3 = to_tensor(image3)
    
    # 对两张图片逐像素相加
    summed_tensor = tensor1 + tensor2

    # 对结果进行归一化
    normalized_tensor = summed_tensor.clamp(0, 1)
    max_pool = torch.nn.MaxPool2d(kernel_size = 7, stride = 1, padding = 3)
    erode_sum = - max_pool(- normalized_tensor)
    
    erode_pred = - max_pool(- tensor3)

    epls = torch.max(erode_sum, erode_pred)

    # 转换回 PIL 图片并保存
    
    output_image = to_pil(epls)
    output_image.save(os.path.join(output_folder, filename))

print("图片处理完成并已保存到输出文件夹。")
