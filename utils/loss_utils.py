#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from utils.general_utils import knn_pcl
# from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from tqdm import tqdm

def l1_loss(render_image, gt, weight=1):
    return torch.abs((render_image - gt) * weight).mean()

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def bce_loss(output, mask=1):
    bce = output * torch.log(output) + (1 - output) * torch.log(1 - output)
    loss = (-bce * mask).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def knn_smooth_loss(gaussian, K):
    xyz = gaussian._xyz
    normal = gaussian.get_normal
    nn_xyz, nn_normal = knn_pcl(xyz, normal, K)
    dist_prj = torch.sum((xyz - nn_xyz) * normal, -1, True).abs()
    loss_prj = dist_prj.mean()

    nn_normal = torch.nn.functional.normalize(nn_normal)
    loss_normal = cos_loss(normal, nn_normal, thrsh=np.pi / 3)
    return loss_prj, loss_normal

def compute_depth_loss(render_depth, gt_depth, mask, tracking=True):
    """ 计算深度损失（Tracking: sum, Mapping: mean）"""
    mask = mask.detach()  # 避免梯度影响
    loss = torch.abs(gt_depth - render_depth)[mask]
    return loss.sum() if tracking else loss.mean()


def compute_rgb_loss(gt_image, render_depth, mask, tracking, use_sil_for_loss, ignore_outlier_depth_loss):
    """
    计算 RGB 颜色损失：
    - Tracking 阶段：使用 `L1 Loss`，可选 `mask` 过滤前景区域。
    - Mapping 阶段：使用 `0.8 * L1 + 0.2 * SSIM` 作为颜色损失。
    """
    if tracking:
        # 仅 Tracking 阶段可能使用 mask 过滤
        if use_sil_for_loss or ignore_outlier_depth_loss:
            color_mask = torch.tile(mask, (3, 1, 1)).detach()  # 扩展 Mask 适用于 RGB
            return torch.abs(render_depth - gt_image)[color_mask].sum()
        return torch.abs(render_depth - gt_image).sum()

    # Mapping 阶段，使用 L1 + SSIM
    return 0.8 * l1_loss(gt_image, render_depth) + 0.2 * (1.0 - ssim(gt_image, render_depth))


def should_continue_tracking(iter, num_iters_tracking, losses_depth, config, do_continue_slam, progress_bar, time_idx):
    """ 判断 Tracking 是否应该继续优化 """
    # 检查是否达到最大迭代次数
    if iter == num_iters_tracking:
        # 1️⃣ 如果深度误差足够小，提前终止
        if losses_depth < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
            return False, iter, num_iters_tracking, do_continue_slam, progress_bar

        # 2️⃣ 启用 do_continue_slam，扩展 Tracking 迭代次数
        elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
            do_continue_slam = True
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            num_iters_tracking = 2 * num_iters_tracking  # 扩展迭代次数
            return True, iter, num_iters_tracking, do_continue_slam, progress_bar

        # 3️⃣ 否则直接终止 Tracking
        else:
            return False, iter, num_iters_tracking, do_continue_slam, progress_bar

    return True, iter, num_iters_tracking, do_continue_slam, progress_bar