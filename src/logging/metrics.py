import math

import cv2
import numpy as np
import torch
import lpips


def tensor2img(tensor):
    """Convert a torch Tensor (C,H,W, [0,1]) to numpy image (H,W,C), [0,255], uint8."""
    tensor = tensor.squeeze().float().cpu().clamp_(0, 1)
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 4:
        n_img = len(tensor)
        from torchvision.utils import make_grid
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    else:
        raise TypeError(f"Only support 3D or 4D tensor, got dim={n_dim}")
    return (img_np * 255.0).round().astype(np.uint8)


def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def compute_psnr(img1, img2):
    """PSNR on [0,255] uint8 images."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(img1, img2):
    """SSIM on [0,255] uint8 images."""
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return _ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(_ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return _ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


_lpips_fn = None


def _get_lpips_fn(device):
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='alex').to(device)
    return _lpips_fn


def compute_lpips(img1_tensor, img2_tensor, device):
    """LPIPS on [0,1] tensors, shape (C,H,W) or (1,C,H,W)."""
    fn = _get_lpips_fn(device)
    if img1_tensor.dim() == 3:
        img1_tensor = img1_tensor.unsqueeze(0)
        img2_tensor = img2_tensor.unsqueeze(0)
    with torch.no_grad():
        return fn(img1_tensor.to(device), img2_tensor.to(device)).item()


def compute_batch_metrics(sr_tensors, gt_tensors, device):
    """Compute PSNR, SSIM, LPIPS for a batch. Returns averages."""
    psnr_list, ssim_list, lpips_list = [], [], []
    batch_size = sr_tensors.shape[0]

    for i in range(batch_size):
        sr_np = tensor2img(sr_tensors[i])
        gt_np = tensor2img(gt_tensors[i])

        psnr_list.append(compute_psnr(sr_np, gt_np))
        ssim_list.append(compute_ssim(sr_np, gt_np))
        lpips_list.append(compute_lpips(sr_tensors[i], gt_tensors[i], device))

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
    }
