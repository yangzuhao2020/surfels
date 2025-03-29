from scene.c3vd import C3VDDataset
import os
from arguments.configs import *
from arguments.c3vd.c3vd_base import config
from utils.image_utils import energy_mask


dataset_config = config["data"]
dataset_config, gradslam_data_cfg = setup_dataset_config(dataset_config)

dataset = C3VDDataset(
        config_dict=gradslam_data_cfg,  # 数据集的配置字典
        basedir=dataset_config["basedir"],  # 数据集的基本目录
        sequence=os.path.basename(dataset_config["sequence"]),  # 数据序列的名称
        start=dataset_config["start"],  # 开始的帧索引
        end=dataset_config["end"],  # 结束的帧索引
        stride=dataset_config["stride"],  # 采样步长（跳过的帧数）
        desired_height=dataset_config["desired_image_height"],  # 目标图像高度
        desired_width=dataset_config["desired_image_width"],  # 目标图像宽度
        device="cuda",  # 运行设备（如 CPU 或 GPU）
        relative_pose=True,  # 让位姿相对于第一帧
        ignore_bad=dataset_config["ignore_bad"],  # 是否忽略损坏的帧
        use_train_split=dataset_config["use_train_split"],  # 是否使用训练集划分
        train_or_test=dataset_config["train_or_test"]  # 选择训练模式或测试模式
    )

gt_rgb, gt_depth, k, gt_pose = dataset[0]
print("gt_rgb:", gt_rgb.device, "gt_depth:", gt_depth.device, "k:", k.device, "gt_pose:", gt_pose.device)
gt_rgb = gt_rgb.permute(2, 0, 1) / 255
gt_depth = gt_depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
mask = (gt_depth > 0) & energy_mask(gt_rgb)
# print("mask:", mask.device)
# print(gt_rgb.device)