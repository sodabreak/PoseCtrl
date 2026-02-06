import os
import re
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# from transformers import CLIPTokenizer
# import cv2

class ResizeAndPad:
    def __init__(self, output_size, fill_color=(255, 255, 255)):
        self.output_size = output_size
        self.fill_color = fill_color

    def __call__(self, image):
        width, height = image.size
        ratio = min(self.output_size / width, self.output_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", (self.output_size, self.output_size), self.fill_color)
        x_offset = (self.output_size - new_width) // 2
        y_offset = (self.output_size - new_height) // 2
        new_image.paste(resized_image, (x_offset, y_offset))
        return new_image

class CombinedDatasetTest(Dataset): 
    def __init__(self, path1=None, path2=None, path3=None, path4=None, path5=None,
                 transform=None, tokenizer=None, txt_subdir_name="txt", pad_to_max_len=True):
        self.transform = transform or transforms.Compose([
            ResizeAndPad(512, fill_color=(255, 255, 255)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_feature = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.samples = []
        self.tokenizer = tokenizer
        self.joints2d_map = {}
        self.txt_subdir_name = txt_subdir_name
        self.pad_to_max_len = pad_to_max_len

        if path1:
            self._load_from_path1(path1)

        v4_style_paths = {"path2": path2, "path3": path3, "path4": path4, "path5": path5}
        for path_name, root_path in v4_style_paths.items():
            if root_path:
                self._load_v4_style_data(root_path, path_name)

        # Limit to first 5 samples for testing
        self.samples = self.samples[:5]

    def _load_name_to_21x3_map(self, txt_path):
        name2arr = {}
        if not os.path.exists(txt_path):
            print(f"[param_txt] {txt_path} not found, will set param_21x3=None for unmatched.")
            return name2arr
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        blocks = re.split(r"\n\s*\n", content)
        for bi, block in enumerate(blocks, 1):
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            name = lines[0]
            rows = []
            for ln in lines[1:]:
                parts = ln.split()
                if len(parts) != 3:
                    continue
                try:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
            if len(rows) != 21:
                print(f"[param_txt][warn] block {bi} name={name}: got {len(rows)} rows (need 21). Skipped.")
                continue
            name2arr[name] = np.array(rows, dtype=np.float32)
        return name2arr

    def _load_points_from_per_image_txt(self, txt_dir, image_stem, image_filename):
        if not txt_dir or (not os.path.exists(txt_dir)):
            print(f"[DEBUG] txt_dir {txt_dir} does not exist or is None")
            return None
        cand1 = os.path.join(txt_dir, f"{image_stem}.txt")
        cand2 = os.path.join(txt_dir, f"{image_filename}.txt")
        use_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
        if use_path is None:
            print(f"[DEBUG] No txt file found for {image_stem} or {image_filename} in {txt_dir}")
            return None
        print(f"[DEBUG] Trying to load points from {use_path}")
        rows = []
        try:
            with open(use_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except UnicodeDecodeError:
            print(f"[DEBUG] UTF-8 decode failed for {use_path}, trying latin-1")
            try:
                with open(use_path, "r", encoding="latin-1") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
            except Exception as e:
                print(f"[DEBUG] Failed to read {use_path}: {e}")
                return None
        if not lines:
            print(f"[DEBUG] No lines in {use_path}")
            return None
        print(f"[DEBUG] Read {len(lines)} lines from {use_path}")
        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) != 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
        if len(rows) == 0:
            print(f"[DEBUG] No valid points in {use_path}")
            return None
        print(f"[DEBUG] Loaded {len(rows)} points from {use_path}")
        return np.array(rows, dtype=np.float32)

    def _read_matrices_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            matrices, matrix = [], []
            for line in lines:
                if 'Capture' not in line:
                    try:
                        row = list(map(float, line.strip().split()))
                        if len(row) == 4:
                            matrix.append(row)
                            if len(matrix) == 4:
                                matrices.append(np.array(matrix))
                                matrix = []
                    except ValueError:
                        continue
            return matrices

    def _load_from_path1(self, root_dir):
        print(f"Loading data from path1: {root_dir}")
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                data_files = sorted(
                    [f for f in os.listdir(folder_path) if f.endswith('.txt')],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
                )
                image_files = sorted(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg')) and f.lower().startswith('capture')],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf')
                )
                feature_file = os.path.join(folder_path, "feature.png")
                if not os.path.exists(feature_file):
                    print(f"Warning: 'feature.png' not found in {folder_path}. Skipping folder.")
                    continue
                projection_matrix_file = None
                view_matrix_file = None
                for data_file in data_files:
                    if 'projectionMatrix' in data_file:
                        projection_matrix_file = os.path.join(folder_path, data_file)
                    elif 'viewMatrix' in data_file:
                        view_matrix_file = os.path.join(folder_path, data_file)
                if projection_matrix_file and view_matrix_file and image_files:
                    projection_matrices = self._read_matrices_from_file(projection_matrix_file)
                    view_matrices = self._read_matrices_from_file(view_matrix_file)
                    if len(projection_matrices) == len(view_matrices) == len(image_files):
                        for proj, view, img_name in zip(projection_matrices, view_matrices, image_files):
                            self.samples.append({
                                'type': 'v1',
                                'image_path': os.path.join(folder_path, img_name),
                                'projection_matrix': proj,
                                'view_matrix': view,
                                'feature_path': feature_file,
                                'text': "highly detailed, anime"
                            })

    def _load_v4_style_data(self, root_dir, path_name):
        print(f"Loading data from {path_name}: {root_dir}")
        param_map_path = os.path.join(root_dir, "image_smpl.txt")
        param_map = self._load_name_to_21x3_map(param_map_path)
        per_image_txt_dir = os.path.join(root_dir, self.txt_subdir_name)
        if not os.path.exists(per_image_txt_dir):
            per_image_txt_dir = None
            print(f"[DEBUG] per_image_txt_dir {per_image_txt_dir} does not exist")
        camera_params_file = os.path.join(root_dir, 'camera_params.txt')
        image_features_file = os.path.join(root_dir, 'image_features.txt')
        joints_file = os.path.join(root_dir, 'merged_joints2d.txt')
        try:
            with open(image_features_file, 'r', encoding='utf-8') as f:
                image_features = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {image_features_file} - {e}")
            image_features = {}
        try:
            with open(camera_params_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: {camera_params_file} not found. Skipping.")
            return
        if os.path.exists(joints_file):
            current_image = None
            with open(joints_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        current_image = line.replace('#', '').strip()
                        self.joints2d_map[current_image] = []
                    else:
                        coords = list(map(float, line.split(',')))
                        self.joints2d_map[current_image].append(coords)
        current_image_path = None
        current_p_matrix, current_v_matrix = [], []
        parsing_mode = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(ext in line for ext in ['.jpg', '.png', '.webp']):
                if current_image_path and len(current_p_matrix) == 4 and len(current_v_matrix) == 4:
                    self._add_v4_sample(root_dir, current_image_path, current_p_matrix, current_v_matrix,
                                        image_features, param_map, per_image_txt_dir)
                current_image_path = line.replace(':', '')
                current_p_matrix, current_v_matrix, parsing_mode = [], [], None
            elif line == 'P:':
                parsing_mode = 'P'
            elif line == 'V:':
                parsing_mode = 'V'
            else:
                try:
                    row_data = list(map(float, re.findall(r'-?\d+\.\d+(?:e-?\d+)?', line)))
                    if len(row_data) == 4:
                        if parsing_mode == 'P' and len(current_p_matrix) < 4:
                            current_p_matrix.append(row_data)
                        elif parsing_mode == 'V' and len(current_v_matrix) < 4:
                            current_v_matrix.append(row_data)
                except ValueError:
                    continue
        if current_image_path and len(current_p_matrix) == 4 and len(current_v_matrix) == 4:
            self._add_v4_sample(root_dir, current_image_path, current_p_matrix, current_v_matrix,
                                image_features, param_map, per_image_txt_dir)

    def _add_v4_sample(self, root_dir, image_path, p_matrix, v_matrix,
                       image_features, param_map, per_image_txt_dir=None):
        image_filename = os.path.basename(image_path)
        image_stem, _ = os.path.splitext(image_filename)
        full_image_path = os.path.join(root_dir, image_filename)
        text_prompt = ", ".join(image_features.get(image_filename, []))
        print(f"[DEBUG] Adding sample for {image_filename}, text: {text_prompt[:50]}...")
        if os.path.exists(full_image_path):
            param_arr = None
            if image_stem in param_map:
                param_arr = param_map[image_stem]
            elif image_filename in param_map:
                param_arr = param_map[image_filename]
            name_txt_arr = self._load_points_from_per_image_txt(per_image_txt_dir, image_stem, image_filename)
            self.samples.append({
                'type': 'v4',
                'image_path': full_image_path,
                'projection_matrix': np.array(p_matrix, dtype=np.float32),
                'view_matrix': np.array(v_matrix, dtype=np.float32),
                'text': text_prompt,
                'param_21x3_np': param_arr,
                'name_txt_np': name_txt_arr
            })
            print(f"[DEBUG] Sample added, points: {name_txt_arr.shape if name_txt_arr is not None else None}")
        else:
            print(f"[DEBUG] Image {full_image_path} does not exist, skipping")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.samples[idx]
        try:
            image = Image.open(sample_data['image_path']).convert('RGB')
        except IOError as e:
            raise IOError(f"Error opening image file '{sample_data['image_path']}': {e}")
        image_tensor = self.transform(image)
        p_matrix_tensor = torch.tensor(sample_data['projection_matrix'], dtype=torch.float32)
        v_matrix_tensor = torch.tensor(sample_data['view_matrix'], dtype=torch.float32)
        text_input_ids = self.tokenizer(
            sample_data['text'],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        final_sample = {
            'image': image_tensor,
            'projection_matrix': p_matrix_tensor,
            'view_matrix': v_matrix_tensor,
            'text': sample_data['text'],
            'type': sample_data['type'],
            'text_input_ids': text_input_ids,
            'image_path': sample_data['image_path']
        }
        if sample_data['type'] == 'v1':
            try:
                feature_image = Image.open(sample_data['feature_path']).convert('RGB')
                final_sample['feature'] = self.transform_feature(feature_image)
            except IOError as e:
                raise IOError(f"Error opening feature file '{sample_data['feature_path']}': {e}")
        if sample_data['type'] == 'v4':
            image_name = os.path.basename(sample_data['image_path'])
            # Skip joints_image creation to avoid cv2 dependency
            # if image_name in self.joints2d_map:
            #     ...
            #     final_sample['joints_image'] = transforms.ToTensor()(joints_img)
            param_np = sample_data.get('param_21x3_np', None)
            final_sample['param_smpl'] = torch.from_numpy(param_np).float() if param_np is not None else None
            name_txt_np = sample_data.get('name_txt_np', None)
            final_sample['points'] = torch.from_numpy(name_txt_np).float() if name_txt_np is not None else None
        return final_sample

if __name__ == "__main__":
    data_root_path_2 = "C:\\Users\\31878\\Desktop\\111\\image_resized"
    txt_subdir_name = "C:\\Users\\31878\\Desktop\\111\\smpl"
    print("Using paths:")
    print("data_root_path_2:", data_root_path_2)
    print("txt_subdir_name:", txt_subdir_name)
    # Skip loading tokenizer and models
    # Use a mock tokenizer
    class MockTokenizer:
        model_max_length = 77
        def __call__(self, text, max_length, padding, truncation, return_tensors):
            # Return a dummy tensor
            import torch
            return {'input_ids': torch.randint(0, 1000, (1, max_length))}
    
    tokenizer = MockTokenizer()
    try:
        train_dataset = CombinedDatasetTest(
            path2=data_root_path_2,
            tokenizer=tokenizer,
            txt_subdir_name=txt_subdir_name
        )
        print(f"Dataset length: {len(train_dataset)}")
        if len(train_dataset) > 0:
            try:
                sample = train_dataset[0]
                print("Sample keys:", list(sample.keys()))
                if 'points' in sample:
                    print("Points shape:", sample['points'].shape if sample['points'] is not None else "None")
                    if sample['points'] is not None:
                        print("Points data (first 5):", sample['points'][:5] if len(sample['points']) > 5 else sample['points'])
                else:
                    print("No 'points' key in sample")
            except Exception as e:
                print("Error getting sample:", str(e))
        else:
            print("Dataset is empty")
    except Exception as e:
        print("Error creating dataset:", str(e))