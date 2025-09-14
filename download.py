import shutil
import kagglehub
import os
current_dir = os.getcwd()

# Download latest version
#path = kagglehub.dataset_download("akrsnv/catdog")
path = kagglehub.dataset_download("mohamedihebhergli/assignment-3-cub200-2011")

target_path = os.path.join(os.getcwd(), "cub200")
shutil.copytree(path, target_path)

print("Dataset copied to:", target_path)