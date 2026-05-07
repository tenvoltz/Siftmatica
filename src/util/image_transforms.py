import numpy as np
import torch
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt


def pil_to_tensor(pil_image: Image.Image, size: Tuple[int, int] = None) -> torch.Tensor:
    if size and pil_image.size != size:
        pil_image = pil_image.resize(size)
    
    pil_image = pil_image.convert('RGB')
    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(img_array).permute(2, 0, 1)


def pil_to_numpy(pil_image: Image.Image, size: Tuple[int, int] = None) -> np.ndarray:
    if size and pil_image.size != size:
        pil_image = pil_image.resize(size)
    
    pil_image = pil_image.convert('RGB')
    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    return img_array.transpose(2, 0, 1)


def load_image_as_tensor(image_path: Union[str, Path], size: Tuple[int, int] = None) -> torch.Tensor:
    pil_image = Image.open(image_path)
    return pil_to_tensor(pil_image, size)


def load_image_as_numpy(image_path: Union[str, Path], size: Tuple[int, int] = None) -> np.ndarray:
    pil_image = Image.open(image_path)
    return pil_to_numpy(pil_image, size)


def ensure_tensor(image: Union[np.ndarray, torch.Tensor], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image.to(dtype)
    return torch.tensor(image, dtype=dtype)


def ensure_numpy(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        return image.numpy()
    return image


def ensure_3channel(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 2:
            return image.unsqueeze(0).repeat(3, 1, 1)
        return image
    else:
        if len(image.shape) == 2:
            return np.repeat(image[np.newaxis, :, :], 3, axis=0)
        return image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    if tensor.shape[0] == 3:
        tensor = np.transpose(tensor, (1, 2, 0))
    
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def tensor_to_numpy_hwc(tensor: torch.Tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        tensor = np.transpose(tensor, (1, 2, 0))

    return np.clip(tensor * 255, 0, 255).astype(np.uint8)

def tensor_to_image(t):
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    t = np.transpose(t, (1, 2, 0))  # CHW -> HWC
    return np.clip(t, 0, 1)


def save_tensor(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    pil_image = tensor_to_pil(tensor)
    pil_image.save(path)


def save_numpy(image: np.ndarray, path: Union[str, Path]) -> None:
    pil_image = numpy_to_pil(image)
    pil_image.save(path)


def display_tensor(tensor: torch.Tensor, title: Optional[str] = None, figsize: Tuple[int, int] = (6, 6)) -> None:
    image = tensor_to_numpy_hwc(tensor)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def display_numpy(image: np.ndarray, title: Optional[str] = None, figsize: Tuple[int, int] = (6, 6)) -> None:
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    display_image = np.clip(image * 255, 0, 255).astype(np.uint8)
    plt.figure(figsize=figsize)
    plt.imshow(display_image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def display_grid(images: list, titles: Optional[list] = None, grid_shape: Optional[Tuple[int, int]] = None, figsize: Tuple[int, int] = (12, 12)) -> None:
    n = len(images)
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        grid_shape = (rows, cols)
    
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = tensor_to_numpy_hwc(img)
        elif isinstance(img, np.ndarray) and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        axes[idx].imshow(img)
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx])
        axes[idx].axis('off')
    
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
