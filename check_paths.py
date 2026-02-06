import os

data_root_path_2 = r"F:\Projects\diffusers\ProgramData\test"
txt_subdir_name = r"F:\Projects\diffusers\ProgramData\new_data\image\smpl"

print("Checking paths...")
print("data_root_path_2 exists:", os.path.exists(data_root_path_2))
print("txt_subdir_name exists:", os.path.exists(txt_subdir_name))

if os.path.exists(data_root_path_2):
    print("Contents of data_root_path_2:", os.listdir(data_root_path_2)[:5])  # first 5 items

if os.path.exists(txt_subdir_name):
    print("Contents of txt_subdir_name:", os.listdir(txt_subdir_name)[:5])  # first 5 items