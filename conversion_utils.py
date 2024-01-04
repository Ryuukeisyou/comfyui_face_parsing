import cv2
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from cv2.typing import MatLike

def image_to_tensor(image: Tensor) -> Tensor:
    typical = image.permute(0, 3, 1, 2).squeeze(0).mul(255).byte()
    return typical

def tensor_to_image(tensor: Tensor) -> Tensor:
    image = tensor.float().div(255).unsqueeze(0).permute(0, 2, 3, 1)
    return image

def tensor_to_cv2BGR(tensor_image: Tensor) -> MatLike: 
    '''
    Converts a typical Tensor image to cv2 BGR image. 
    '''
    # Step 1: Convert PyTorch tensor to NumPy array
    numpy_image = tensor_image.permute(1, 2, 0).numpy()

    # Step 2: Convert NumPy array to cv2 image
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return cv2_image

def tensor_to_cv2YCrCb(tensor_image: Tensor) -> MatLike:
    '''
    Converts a typical Tensor image to cv2 YCrCb image. 
    '''
    # Step 1: Convert PyTorch tensor to NumPy array
    numpy_image = tensor_image.permute(1, 2, 0).numpy()

    # Step 2: Convert NumPy array to cv2 image
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2YCrCb)

    return cv2_image

def tensor_to_cv2HSV(tensor_image: Tensor) -> MatLike:
    '''
    Converts a typical Tensor image to cv2 YCrCb image. 
    '''
    # Step 1: Convert PyTorch tensor to NumPy array
    numpy_image = tensor_image.permute(1, 2, 0).numpy()

    # Step 2: Convert NumPy array to cv2 image
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2HSV)

    return cv2_image

def cv2BGR_to_tensor(cv2_image) -> Tensor:
    '''
    Converts a cv2 image to typical Tensor image.
    '''
    # Step 1: Convert cv2 image to NumPy array
    # If the image is in BGR order, convert it to RGB
    numpy_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert NumPy array to PyTorch tensor
    tensor_image = torch.tensor(numpy_image)

    # Permute dimensions to match PyTorch tensor format (3, H, W)
    tensor_image = tensor_image.permute(2, 0, 1)
    return tensor_image