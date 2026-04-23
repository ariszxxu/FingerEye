import numpy as np
from omegaconf import ListConfig
import os
import random           
import numpy as np      
import torch
import warp as wp
import cv2


def slice_with_list(full_array, slice_list, dim_to_slice=-1):
    """
    Slice an array along `dim_to_slice` using a slice_list that may contain:
      - single indices: int
      - ranges: [start, end] (end is exclusive, like Python slicing)

    Args:
        full_array: np.ndarray
        slice_list: list of int or [start, end]
        dim_to_slice: dimension to slice along

    Returns:
        Sliced array with concatenated results
    """

    slices = []
    for item in slice_list:
        if isinstance(item, int):
            # Single index -> keep shape along that dim (unsqueeze)
            slc = full_array.take(indices=[item], axis=dim_to_slice)
        elif isinstance(item, (list, tuple, ListConfig)) and len(item) == 2:
            start, end = item
            slc = full_array[(slice(None),) * dim_to_slice + (slice(start, end),)]
        else:
            raise ValueError(f"Unsupported slice spec: {item}")
        slices.append(slc)

    return np.concatenate(slices, axis=dim_to_slice)


def configure_seed(seed: int | None, torch_deterministic: bool = False) -> int:
    """Set seed across all random number generators (torch, numpy, random, warp).

    Args:
        seed: The random seed value. If None, generates a random seed.
        torch_deterministic: If True, enables deterministic mode for torch operations.

    Returns:
        The seed value that was set.
    """
    if seed is None or seed == -1:
        seed = 42 if torch_deterministic else random.randint(0, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wp.rand_init(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

def draw_status_overlay(img, status):
    """
    Draws a thick border and large status text on the image.
    img: Numpy array (H, W, 3) assumed to be uint8
    status: 'success' or 'fail'
    """
    # Make a writeable copy to ensure we don't error on torch-converted tensors
    img = img.copy() 
    h, w, _ = img.shape
    
    # Colors (Assuming RGB)
    # Green for Success, Red for Failure
    color = (0, 255, 0) if status == 'success' else (255, 0, 0)
    text = "SUCCESS" if status == 'success' else "FAIL"
    
    # 1. Draw Thick Border
    thickness = 10  # Increased thickness
    # Top, Bottom, Left, Right
    img[:thickness, :] = color
    img[-thickness:, :] = color
    img[:, :thickness] = color
    img[:, -thickness:] = color
    
    # 2. Add Text (using OpenCV)
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2      # Increased scale
    font_thickness = 3    # Thicker line weight for readability
    
    # Position text slightly inward so it clears the thick border
    text_x = 20
    text_y = 50 
    
    # Draw Text (Color) - No Shadow
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness)

    return img
