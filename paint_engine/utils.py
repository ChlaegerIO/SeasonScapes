import os
import numpy as np
import torch
from PIL import Image
import imageio
import cv2


def pil2tensor(pil_img, device=torch.device('cpu')):
    """Convert PIL image to normalized tensor in [0, 1] range."""
    np_img = np.array(pil_img)
    if np_img.ndim == 2:  # Grayscale image
        np_img = np_img[:, :, np.newaxis]
    tensor_chw = torch.Tensor(np_img).to(device).permute(2, 0, 1) / 255.0
    return tensor_chw.unsqueeze(0)


def save_tensor_image(tensor: torch.Tensor, save_path: str):
    """Save tensor image to disk, image in [0, 1] range.
    Tensor dimensions: [1, c, h, w], [3, h, w], [1, h, w]
    """
    if len(os.path.dirname(save_path)) > 0 and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # [1, c, h, w]-->[c, h, w]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)  
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()  # [c, h, w]-->[h, w, c]
    Image.fromarray((tensor * 255).astype(np.uint8)).save(save_path)

def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor[0].permute(1, 2, 0).contiguous().clamp(0, 1).detach()  # [N, C, H, W ]-->[C, H, W]-->[H, W, C]
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def save_video(video_frames, save_path):
    imageio.mimwrite(save_path, video_frames, fps=25, quality=8, macro_block_size=1)


def extract_bg_mask(img_path, mask_color=[0, 0, 255], dilate_kernel=3):
    """
    From an image, extract the background mask based on a specific color.
    :param mask_color:  BGR
    :return:
    """
    img = cv2.imread(img_path)

    mask = (img == mask_color).all(axis=2).astype(np.float32)
    mask = mask[:,:,np.newaxis]

    if dilate_kernel > 0:
        mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))[:,:,np.newaxis]
    mask = (mask * 255).astype(np.uint8)
    return mask