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

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch, quaternion2rotmatrix, rotmatrix2quaternion
from utils.graphics_utils import fov2focal
from utils.image_utils import resize_image
import torch.nn.functional as F
WARNED = False

def loadCam(args, id, cam_info, resolution_scale, scene_scale, camera_lr):
    orig_w, orig_h = cam_info.image.size
    if args.resolution == 1:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_mask = resize_image(cam_info.mask, [resolution[1], resolution[0]])
    resized_mono = None if cam_info.mono is None else resize_image(cam_info.mono, [resolution[1], resolution[0]])

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    return Camera(colmap_id=cam_info.uid,
                  R=cam_info.R, 
                  T=cam_info.T, 
                  FoVx=cam_info.FovX,
                  FoVy=cam_info.FovY, 
                  prcppoint=cam_info.prcppoint,
                  image=gt_image, 
                  gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, 
                  uid=id, 
                  data_device=args.data_device,
                  mask=resized_mask, 
                  mono=resized_mono, 
                  scene_scale=scene_scale, 
                  camera_lr=camera_lr)
    

def cameraList_from_camInfos(cameras, resolution_scale, cameras_extent, camera_lr, args, time_idx, camlist):
    camlist[resolution_scale].append(loadCam(args, time_idx, cameras, resolution_scale, cameras_extent, camera_lr))
    if time_idx > 0:
        initialize_camera_pose(camlist[resolution_scale], time_idx)
    return camlist[resolution_scale]

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width),
        'prcp': camera.prcppoint.tolist()
    }
    return camera_entry

def interpolate_camera(cam_lst, num):
    cam_inter = []
    count = 0
    for i in range(len(cam_lst) - 1):
        c0 = cam_lst[i]
        c0.image_name = str(count).zfill(5)
        count += 1
        cam_inter.append(c0)
        c1 = cam_lst[i + 1]
        q0 = rotmatrix2quaternion(c0.R[None], True)
        q1 = rotmatrix2quaternion(c1.R[None], True)
        # img = torch.zeros_like(c0.original_image)
        for j in range(1, num):
            k = 1 - j / num
            t = k * c0.T + (1 - k) * c1.T
            q = k * q0 + (1 - k) * q1
            R = quaternion2rotmatrix(torch.nn.functional.normalize(q))[0]
            fovx = k * c0.FoVx + (1 - k) * c1.FoVx
            fovy = k * c0.FoVy + (1 - k) * c1.FoVy
            prcp = k * c0.prcppoint + (1 - k) * c1.prcppoint
            c = Camera(None, R.cpu().numpy(), t.cpu().numpy(), fovx, fovy, prcp.numpy(), image_name=str(count).zfill(5),
                       img_w=c0.original_image.shape[2], img_h=c0.original_image.shape[1])
            count += 1
            cam_inter.append(c)
    cam_last = cam_lst[-1]
    cam_last.image_name = str(count).zfill(5)
    cam_inter.append(cam_last)
    return cam_inter


def initialize_camera_pose(camlist, time_idx, forward_prop = True):
    """初始化当前帧的相机位姿，可以使用恒定模型，或者是直接的复制上面一帧率。"""
    with torch.no_grad():
        if time_idx > 1 and forward_prop:
            # Rotation
            prev_rot1 = camlist[time_idx-1].q.detach()  # 上一帧的旋转 (1, 4)
            prev_rot2 = camlist[time_idx-2].q.detach()  # 上上帧的旋转 (1, 4)

            # 使用 Slerp 外推：从 prev_rot2 到 prev_rot1 的旋转速度，应用到 prev_rot1
            relative_rot = slerp(prev_rot2, prev_rot1, t=1.0)  # t=1 表示完整步长
            new_rot = slerp(prev_rot1, relative_rot, t=2.0)  # t=2 表示外推一步
            camlist[time_idx].q = new_rot
            
            # Translation
            prev_tran1 = camlist[time_idx-1].T.detach()
            prev_tran2 = camlist[time_idx-2].T.detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            camlist[time_idx].T = new_tran
            # 新的一帧的位置。
            
        else:
            # Initialize the camera pose for the current frame
            camlist[time_idx].q = camlist[time_idx - 1].q.detach()
            camlist[time_idx].T = camlist[time_idx - 1].T.detach()
            # 直接复制上一帧的位姿。
    return camlist


def slerp(q0, q1, t, eps=1e-8):
    """
    球面线性插值 (Slerp) 两个四元数，确保输出始终是单位四元数。
    Args:
        q0: 起始四元数，形状 (..., 4)
        q1: 目标四元数，形状 (..., 4)
        t: 插值比例，标量或张量（0 <= t <= 1 表示插值，t > 1 表示外推）
        eps: 避免除零的小值
    Returns:
        插值后的单位四元数，形状与 q0, q1 相同
    """
    # 确保输入四元数是单位向量
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)

    # 计算点积
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)  # (..., 1)

    # 如果点积为负，反转 q1 以走短路径
    mask = dot < 0
    q1 = torch.where(mask, -q1, q1)
    dot = torch.where(mask, -dot, dot)

    # 处理边界条件
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)  # 避免 acos 的数值问题
    theta_0 = torch.acos(dot)  # 夹角 (..., 1)
    sin_theta_0 = torch.sin(theta_0)  # sin(夹角)

    # 如果 sin_theta_0 接近零（即 q0 ≈ q1 或 q0 ≈ -q1），退化为线性插值
    if torch.all(sin_theta_0 < eps):
        qt = (1 - t) * q0 + t * q1
        return F.normalize(qt, dim=-1)

    # 计算插值角度和系数
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / (sin_theta_0 + eps)
    s1 = sin_theta_t / (sin_theta_0 + eps)

    # 插值
    qt = s0 * q0 + s1 * q1

    # 强制归一化，确保输出是单位四元数
    qt = F.normalize(qt, dim=-1)
    return qt