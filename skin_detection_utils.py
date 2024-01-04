import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)
from torch import Tensor
import torch
import cv2
import numpy as np
from numpy import ndarray
import conversion_utils
import torchvision
import matplotlib.pyplot as plt


def detect_skin_RGB(image: Tensor, kernel_size: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    '''
    Detect typical Asian skin using RGB method.
    '''
    rs = image[0]
    gs = image[1]
    bs = image[2]
    maxs = torch.max(image, 0)[0]
    mins = torch.min(image, 0)[0]

    # Typical Asian skin color should meet the following constraints.

    # In uniform illumination conditions:
    # r>95 and g>40 and b>20 and max(r,g,b)-min(r,g,b)>15 and r-g>15 and r>b
    uniform_illumination_mask : Tensor = (rs > 95) & (gs > 40) & (bs > 20) & (rs - gs > 15) & (rs > bs) & (maxs - mins > 15)

    # In lateral illumination conditions:
    # r>220 and g>210 and b>170 and abs(r-g)<=15 and r>b and g>b
    lateral_illumination_mask : Tensor = (rs > 220) & (gs > 210) & (bs > 170) & (rs > bs) & (gs > bs) & (torch.abs(rs - gs) <= 15) 

    skin_mask = uniform_illumination_mask | lateral_illumination_mask

    uniform_illumination_result : ndarray = uniform_illumination_mask.numpy().astype(np.uint8)
    lateral_illumination_result : ndarray = lateral_illumination_mask.numpy().astype(np.uint8)
    skin_result : ndarray = skin_mask.numpy().astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    uniform_illumination_result = cv2.morphologyEx(uniform_illumination_result, cv2.MORPH_OPEN, kernel)
    lateral_illumination_result = cv2.morphologyEx(lateral_illumination_result, cv2.MORPH_OPEN, kernel)
    skin_result = cv2.morphologyEx(skin_result, cv2.MORPH_OPEN, kernel)

    return torch.tensor(uniform_illumination_mask), torch.tensor(lateral_illumination_mask), torch.tensor(skin_mask)

def detect_skin_YcrCb(image: Tensor, kernel_size: int = 0) -> Tensor:
    '''
    Detect typical Asian skin using YCrCb method.
    '''
    ycrcb = conversion_utils.tensor_to_cv2YCrCb(image)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    mask = (133 <= cr) & (cr <= 173) & (77 <= cb) & (cb <= 127) # type: ignore
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    result_tensor = torch.tensor(result)
    return result_tensor

def detect_skin_HSV(image: Tensor, kernel_size: int = 0) -> Tensor:
    '''
    Detect typical Asian skin using HSV method.
    '''
    hsv = conversion_utils.tensor_to_cv2HSV(image)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (7 <= h) & (h <= 35) & (s >= 23) & (v >= 90) # type: ignore
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    result_tensor = torch.tensor(result)
    return result_tensor

def detect_skin_inference(image: Tensor) -> Tensor:
    raise NotImplementedError

if __name__ == '__main__':
    # Test only codes here
    cur_dir = os.path.dirname(__file__)
    image_path = os.path.join(cur_dir, '../resources/original.jpg')
    image = torchvision.io.read_image(image_path)
    plt.imshow(image.permute(1, 2, 0))

    mask_rgb = detect_skin_RGB(image)
    plt.imshow(mask_rgb[0])
    plt.imshow(mask_rgb[1])
    plt.imshow(mask_rgb[2])

    mask_ycrcb = detect_skin_YcrCb(image)
    plt.imshow(mask_ycrcb)

    mask_hsv = detect_skin_HSV(image)
    plt.imshow(mask_hsv)

    a = 0
    
