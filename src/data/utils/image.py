import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple
import tensorflow as tf
from src.data.utils.augmentations import augment_image

def process_image(image: torch.Tensor) -> torch.Tensor:
    """Process the image by rounding, clipping, and casting to uint8."""
    # Ensure the image is in float32 before processing (if it's in a different dtype)
    image = image.float()

    # Round the image values to nearest integer
    image = torch.round(image)

    # Clip the values to ensure they are within the range [0, 255]
    image = torch.clamp(image, 0, 255)

    # Convert the image back to uint8 type
    image = image.to(torch.uint8)

    return image

def resize_image_tensor(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize a tensor image using bilinear interpolation (or other methods)."""
    # Ensure the image is in the correct shape [batch_size, channels, height, width]
    image = process_image(image)
    image = image.unsqueeze(0)  # Add batch dimension if needed
    resized_image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
    return resized_image.squeeze(0)  # Remove the batch dimension


def read_resize_encode_image_pytorch(path: str, size: Tuple[int, int]) -> bytes:
    """Reads, decodes, resizes, and then re-encodes an image using PyTorch."""
    image = Image.open(path)
    
    resize_transform = transforms.Resize(size)
    image = resize_transform(image)
    
    image = image.convert("RGB")
    image = torch.tensor(np.array(image), dtype=torch.float32)
    image = torch.round(image)
    image = torch.clamp(image, 0, 255)
    image = image.to(torch.uint8)

    return image

def concatenate_images(left_image: str, middle_image: str, right_image: str):
    middle_img = Image.open(path).convert("RGB")
    left_img = Image.open(path).convert("RGB")
    right_img = Image.open(path).convert("RGB")

    middle_img = middle_img[:, 200:1400, :] # (H, W, C)
    left_img = left_img[:, :1400, :] # (H, W, C)
    right_img = right_img[:, 200:, :] # (H, W, C)
    rgb = np.concatenate((front_left_img, front_img, front_right_img), axis=1)
    return rgb


def image_normalization(path: str, size: Tuple[int, int]):
    image = read_resize_encode_image_pytorch(path, size)
    np_array = image.numpy()
    tf_tensor = tf.convert_to_tensor(np_array)
    image_augment_kwargs = dict(random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1],),
                                random_brightness=[0.1],
                                random_contrast=[0.9, 1.1],
                                random_saturation=[0.9, 1.1],
                                random_hue=[0.05],
                                augment_order=[
                                    "random_resized_crop",
                                    "random_brightness",
                                    "random_contrast",
                                    "random_saturation",
                                    "random_hue",
                                ],
                            )
    output = augment_image(image=tf_tensor, **image_augment_kwargs)
    np_image = output.numpy().astype(np.uint8)
    image = torch.as_tensor(np_image)
    return image