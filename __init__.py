import os
import folder_paths
from torchvision.datasets.utils import download_url

# Download files.

models_path = folder_paths.models_dir
face_parsing_path = os.path.join(models_path, "face_parsing")
ultralytics_bbox_path = os.path.join(models_path, "ultralytics", "bbox")

if not os.path.exists(face_parsing_path):
    os.makedirs(face_parsing_path)
if not os.path.exists(ultralytics_bbox_path):
    os.makedirs(ultralytics_bbox_path)
folder_paths.add_model_folder_path("ultralytics_bbox", ultralytics_bbox_path)
if not os.path.exists(os.path.join(face_parsing_path, "model.safetensors")):
    download_url("https://huggingface.co/jonathandinu/face-parsing/resolve/main/model.safetensors?download=true", face_parsing_path, "model.safetensors")
if not os.path.exists(os.path.join(face_parsing_path, "config.json")):
    download_url("https://huggingface.co/jonathandinu/face-parsing/resolve/main/config.json?download=true", face_parsing_path, "config.json")
if not os.path.exists(os.path.join(face_parsing_path, "preprocessor_config.json")):
    download_url("https://huggingface.co/jonathandinu/face-parsing/resolve/main/preprocessor_config.json?download=true", face_parsing_path, "preprocessor_config.json")
if not os.path.exists(os.path.join(ultralytics_bbox_path, "face_yolov8m.pt")):
    download_url("https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt", ultralytics_bbox_path)


# Install packages.
    
import subprocess
import pkg_resources

cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir, 'requirements.txt')) as f:
    required_packages = f.read().splitlines()
    installed_packages = [pkg.key for pkg in pkg_resources.working_set]
    missing_packages = set(required_packages) - set(installed_packages)
    if missing_packages:
        subprocess.check_call(["pip", "install", *missing_packages])


# Export classes.

from .face_parsing_nodes import NODE_CLASS_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS']