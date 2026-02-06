import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'poseCtrl'))

from poseCtrl.data.dataset import CustomDataset, load_base_points
import torch

# 测试数据集读取
def test_dataset_loading():
    # 假设数据集路径是 dataSet/ 或类似
    dataset_path = 'dataSet'  # 根据工作区结构调整

    try:
        dataset = CustomDataset(root_dir=dataset_path)
        print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")

        # 测试获取一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print("Sample keys:", list(sample.keys()))
            print("Image shape:", sample['image'].shape)
            print("Feature shape:", sample['feature'].shape)
            print("Projection matrix shape:", sample['projection_matrix'].shape)
            print("View matrix shape:", sample['view_matrix'].shape)

            # 检查是否有points，如果没有，尝试加载base points
            if 'points' not in sample:
                print("Points not found in sample. Loading base points...")
                base_points_path = os.path.join(dataset_path, 'standardVertex.txt')
                if os.path.exists(base_points_path):
                    base_points = load_base_points(base_points_path)
                    print("Base points loaded. Shape:", base_points.shape)
                else:
                    print("Base points file not found at:", base_points_path)
            else:
                print("Points found in sample. Shape:", sample['points'].shape)
        else:
            print("Dataset is empty.")

    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    test_dataset_loading()
