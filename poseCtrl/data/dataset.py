import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
from matplotlib import pyplot as plt
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and txt files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor(), 
        ])
        self.samples = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                data_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg')) and f.lower().startswith('capture')]
                feature_file = os.path.join(folder_path, "feature.png")
                if not os.path.exists(feature_file):
                    raise FileNotFoundError(f"'{feature_file}' does not exist, please check again.")
                if len(data_files) == 134 and len(image_files) == 132:
                    projection_matrix_file = None
                    view_matrix_file = None
                    for data_file in data_files:
                        if 'projectionMatrix' in data_file:
                            projection_matrix_file = os.path.join(folder_path, data_file)
                        elif 'viewMatrix' in data_file:
                            view_matrix_file = os.path.join(folder_path, data_file)
                    image_files = [os.path.join(folder_path, img) for img in image_files]
                    if projection_matrix_file and view_matrix_file and image_files:
                        # 修改为每个图片与对应的矩阵文件配对
                        projection_matrices = self.read_matrices(projection_matrix_file)
                        view_matrices = self.read_matrices(view_matrix_file)
                        self.samples.extend([(proj, view, img, feature_file) for proj, view, img in zip(projection_matrices, view_matrices, image_files)])
                        # 添加调试信息
                        # print(f"Folder: {folder_name}, Number of images: {len(image_files)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        projection_matrix, view_matrix, image_file, feature_file = self.samples[idx]

        # 确保图片文件路径正确
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")

        # 读取图像
        try:
            image = Image.open(image_file).convert('RGB')
        except IOError as e:
            raise IOError(f"Error opening image file {image_file}: {e}")
        
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Image file not found: {feature_file}")

        # 读取图像
        try:
            feature = Image.open(feature_file).convert('RGB')
        except IOError as e:
            raise IOError(f"Error opening image file {feature_file}: {e}")


        # 处理图像
        image = self.transform(image)  # **确保转换成 Tensor**
        feature = self.transform(feature)
        # 确保矩阵是 Tensor
        projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
        view_matrix = torch.tensor(view_matrix, dtype=torch.float32)

        # 确保 projection_matrix 和 view_matrix 形状正确
        if projection_matrix.shape != (4, 4):
            raise ValueError(f"Projection matrix shape is incorrect: {projection_matrix.shape}")
        if view_matrix.shape != (4, 4):
            raise ValueError(f"View matrix shape is incorrect: {view_matrix.shape}")

        sample = {
            'image': image,
            'projection_matrix': projection_matrix,
            'view_matrix': view_matrix,
            'feature': feature
        }
        return sample

    def read_matrices(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            matrices = []
            matrix = []
            for line in lines:
                if 'Capture' not in line:  # 跳过包含 'Capture' 的行
                    try:
                        row = list(map(float, line.strip().split()))
                        if len(row) == 4:  # 确保每一行有4个元素
                            matrix.append(row)
                            if len(matrix) == 4:  # 确保矩阵有4行
                                matrices.append(np.array(matrix))
                                matrix = []
                    except ValueError:
                        pass
            return matrices
        

def load_base_points(path):
    points = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' does not exist, please check again.")

    if path.endswith('.txt'):
        with open(path, 'r') as f:
            lines = f.readlines()  
            num_lines = len(lines)  
            if num_lines < 13860:
                for line in lines:
                    coords = list(map(float, line.strip().split()))
                    if len(coords) != 4:
                        raise ValueError(f"All points should have 4 coordinates, but found {len(coords)} in: {coords}")
                    points.append(coords)

                missing_points = 13860 - num_lines
                points.extend([[0, 0, 0, 0]] * missing_points)
            else:
                # 只取前 13860 个点
                for line in lines[:13860]:
                    coords = list(map(float, line.strip().split()))
                    if len(coords) != 4:
                        raise ValueError(f"All points should have 4 coordinates, but found {len(coords)} in: {coords}")
                    points.append(coords)

        points_tensor = torch.tensor(np.array(points, dtype=np.float32))
        return points_tensor

    else:
        pass  

""" add 'set PYTHONPATH=F:/Projects/diffusers/Project' """
# train_dataset = CustomDataset("F:\\Projects\\diffusers\\ProgramData\\sample_new")

# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset,
#     shuffle=True,
#     batch_size=32,
# )
# print(len(train_dataset))
# path=r'F:\Projects\diffusers\Project\PoseCtrl\dataSet\standardVertex.txt'
# base_points=load_base_points(path)
# print(base_points.shape)
