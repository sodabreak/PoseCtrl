import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir, "PoseCtrl")
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir,"poseCtrl"))

from poseCtrl.data.dataset import CombinedDatasetTest
import torch

class FakeTokenizer:
    def __init__(self):
        self.model_max_length = 77
    
    def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
        # 返回假的input_ids
        return {'input_ids': torch.randint(0, 1000, (1, self.model_max_length))}

tokenizer = FakeTokenizer()

print("Checking paths...")
path2 = r"C:\Users\31878\Desktop\111\image_resized"
txt_dir = r"C:\Users\31878\Desktop\111\smpl"

print(f"path2 exists: {os.path.exists(path2)}")
print(f"txt_dir exists: {os.path.exists(txt_dir)}")

if os.path.exists(path2):
    print(f"Contents of path2: {os.listdir(path2)[:5]}")  # 只显示前5个

if os.path.exists(txt_dir):
    print(f"Contents of txt_dir: {os.listdir(txt_dir)[:5]}")  # 只显示前5个

print("Starting dataset creation...")
try:
    train_dataset = CombinedDatasetTest(
        path2=path2,
        tokenizer=tokenizer,
        txt_subdir_name=txt_dir
    )
    print(f"Dataset loaded successfully with {len(train_dataset)} samples")

    # 尝试读取第一个样本
    if len(train_dataset) > 0:
        print("Reading first sample...")
        sample = train_dataset[0]
        print("Sample keys:", list(sample.keys()))
        if 'points' in sample:
            points = sample['points']
            if points is not None:
                print(f"Points shape: {points.shape}")
                print(f"Points data: {points}")
            else:
                print("Points is None")
        else:
            print("No 'points' key in sample")
    else:
        print("No samples in dataset")

except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()