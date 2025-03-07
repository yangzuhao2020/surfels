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

import os
import torch
from random import randint
from utils.loss_utils import *
from utils.keyframe_selection import *
from gaussian_renderer import render
import numpy as np
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, match_depth, normal2curv, resize_image, cross_sample
from utils.general_utils import build_rotation
from torchvision.utils import save_image
from argparse import ArgumentParser, Namespace
import time
import os
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.c3vd import C3VD
from arguments.configs import *

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(args, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from,config):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(args)
    config = setup_config_defaults(config)
    output_dir, eval_dir = setup_directories(config) # 实验结果输出文件。
    dataset_config = config["data"]
    dataset_config, gradslam_data_cfg = setup_dataset_config(dataset_config)
    dataset = C3VD(
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
    use_mask = args.use_mask
    scene = Scene(args, gaussians, dataset, opt.camera_lr, shuffle=False)
    gaussian.initialize_first_timestep()
    gaussians.training_setup(opt) # 对模型参数进行初始化、优化器设置以及学习率调度器配置。
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif use_mask: # visual hull init
        gaussians.mask_prune(scene.getTrainCameras(), 4) # 掩码剪枝
        None

    opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))
    # 调整密度的周期。
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    background = torch.rand((3), dtype=torch.float32, device="cuda") if args.random_background else background
    patch_size = [float('inf'), float('inf')] # 设置两个无穷大的元素。
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    pool = torch.nn.MaxPool2d(9, stride=1, padding=4)
    # 在不改变特征图尺寸的情况下，提取局部区域的最大值

    viewpoint_stack = None # 记录相机视角
    ema_loss_for_log = 0.0

    checkpoint_time_idx = 0
    total_num_frames = None
    config = {}
    curr_data = {}
    keyframe_list = []
    gt_w2c_all_frames = []
    actural_keyframe_ids = []
    
    for time_idx in tqdm(range(checkpoint_time_idx, total_num_frames)):
        gt_rgb, gt_depth, _, gt_pose = args[time_idx]
        iter_start.record()
        gaussians.update_learning_rate(time_idx) # 传入迭代次数，修改位置参数的学习率。
        gt_rgb = gt_rgb.permute(2, 0, 1) / 255
        gt_depth = gt_depth.permute(2, 0, 1)
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()[:]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            tracking = True
            gaussians.initialize_optimizer(config['tracking']['lrs'], mode="tracking")
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = gaussians._cam_rots[..., time_idx].detach().clone()
            candidate_cam_tran = gaussians._cam_trans[..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            num_iters_tracking_cam = config['tracking']['num_iters'] # 相机优化的次数。
            num_iters_mapping = config['mapping']['num_iters'] # 高斯点的优化，每帧优化的次数。
            progress_bar = tqdm(range(num_iters_tracking_cam), desc=f"Tracking Time Step: {time_idx}")
            iter = 0
            do_continue_slam = False
            
            while True:
                if (time_idx - 1) == debug_from:
                    pipe.debug = True
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, patch_size)
                render_image, render_normal, render_depth, render_opac, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
                render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                loss,loss_dict,mask_vis= get_loss(viewpoint_cam, gaussians, pipe, 
                                                background, pool, tracking,
                                                patch_size, curr_data, use_mask=True)
                loss.backward() # 反向传播。
                iter_end.record()
                gaussians.optimizer.step() # 更新参数
                gaussians.optimizer.zero_grad() # 清空梯度
                with torch.no_grad():
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = gaussians._cam_rots[..., time_idx].detach().clone()
                        candidate_cam_tran = gaussians._cam_trans[..., time_idx].detach().clone()
                    progress_bar.update(1)
                    
                iter += 1
                tracking_active, iter, num_iters_tracking_cam, do_continue_slam, progress_bar = should_continue_tracking(
                    iter, num_iters_tracking_cam, 
                    loss_dict["depth"], config, do_continue_slam,
                    progress_bar, time_idx)

                if not tracking_active:
                    break  # 终止 Tracking 过程
            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                gaussians._cam_rots[..., time_idx] = candidate_cam_unnorm_rot
                gaussians._cam_trans[..., time_idx] = candidate_cam_tran
                
                
        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            variables = gaussians.add_new_gs()
            if not config['distance_keyframe_selection']:
                with torch.no_grad():
                    # 1️⃣ 归一化当前帧的相机旋转
                    curr_cam_rot = F.normalize(gaussians._cam_rots[..., time_idx].detach())
                    curr_cam_tran = gaussians._cam_trans[..., time_idx].detach()
                    # 2️⃣ 构造当前帧的世界到相机变换矩阵 (w2c)
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # 3️⃣ 选择最相关的关键帧
                    num_keyframes = config['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(gt_depth, curr_w2c, curr_cam_rot["intrinsics"], 
                                                                    keyframe_list[:-1], num_keyframes)
                    # 4️⃣ 获取关键帧时间索引
                    selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    # 5️⃣ 确保最后一个关键帧被选择
                    if len(keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(keyframe_list[-1]['id'])
                        selected_keyframes.append(len(keyframe_list)-1)
                    # 6️⃣ 添加当前帧到关键帧列表
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # 7️⃣ 打印最终选择的关键帧
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            gaussians.update_learning_rate(time_idx) # 传入迭代次数，修改位置参数的学习率。
            # mapping
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                if not config['distance_keyframe_selection']:
                    # Randomly select a frame until current time step amongst keyframes
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    actural_keyframe_ids.append(selected_rand_keyframe_idx)
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = gt_rgb
                        iter_depth = gt_depth
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                
                else:
                    if len(actural_keyframe_ids) == 0:
                        if len(keyframe_list) > 0:
                            curr_position = params['cam_trans'][..., time_idx].detach().cpu()
                            actural_keyframe_ids = keyframe_selection_distance(time_idx, curr_position, keyframe_list, config['distance_current_frame_prob'], num_iters_mapping)
                        else:
                            actural_keyframe_ids = [0] * num_iters_mapping
                        print(f"\nUsed Frames for mapping at Frame {time_idx}: {[keyframe_list[i]['id'] if i != len(keyframe_list) else 'curr' for i in actural_keyframe_ids]}")

                    selected_keyframe_ids = actural_keyframe_ids[iter]

                    if selected_keyframe_ids == len(keyframe_list):
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = gt_rgb
                        iter_depth = gt_depth
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_keyframe_ids]['id']
                        iter_color = keyframe_list[selected_keyframe_ids]['color']
                        iter_depth = keyframe_list[selected_keyframe_ids]['depth']
            if (time_idx - 1) == debug_from:
                    pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, patch_size)
            render_image, render_normal, render_depth, render_opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            loss,loss_dict,mask_vis= get_loss(viewpoint_cam, gaussians, pipe, 
                                            background, pool, tracking,
                                            patch_size, curr_data, use_mask=True)
            loss.backward() # 反向传播。
            
            with torch.no_grad():
                if (time_idx in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(time_idx))
                    scene.save(time_idx)

                # Densification
                if time_idx > opt.densify_from_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    # 记录最大的投影半径。
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    min_opac = 0.1
                    if time_idx % opt.densification_interval == 0:
                        gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                        gaussians.adaptive_densify(opt.densify_grad_threshold, # 密度增强的梯度阈值。 这里可能会对误差大的地方加点。
                                                scene.cameras_extent) 
                    
                if (time_idx - 1) % 1000 == 0:
                    normal_wrt = normal2rgb(render_normal, mask_vis)
                    depth_wrt = depth2rgb(render_depth, mask_vis)
                    img_wrt = torch.cat([curr_data["image"], render_image, normal_wrt * render_opac, depth_wrt * render_opac], 2)
                    # 图片拼接
                    save_image(img_wrt.cpu(), f'test/test.png')
                
                if time_idx < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad()
                    # viewpoint_cam.optimizer.step()
                    # viewpoint_cam.optimizer.zero_grad()

                if (time_idx in checkpoint_iterations):
                    # gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                    print("\n[ITER {}] Saving Checkpoint".format(time_idx))
                    torch.save((gaussians.capture(), time_idx), scene.model_path + "/chkpnt" + str(time_idx) + ".pth")

def get_loss(render_pkg, viewpoint_cam, gaussians, 
             background, pool, tracking,
             curr_data, use_mask=True):
    # loss_depth_normal
    mask_gt = viewpoint_cam.get_gtMask(use_mask) # (1, H, W) 全1的张量。
    # gt_image = viewpoint_cam.get_gtImage(background, use_mask)
    mask_vis = (render_pkg["opac"].detach() > 1e-5)
    # 确定哪些像素是背景，要求把不透明度很低的点找出来，那么它对应的区域说明是不存在物体。
    loss_mask = (render_pkg["opac"] * (1 - pool(mask_gt))).mean()
    # 为什么会出现池化这个操作，原因很简单，他想要扩大边界一点。
    render_pkg["normal"] = torch.nn.functional.normalize(render_pkg["normal"], dim=0) * mask_vis
    # 归一化，然后将区域对应背景的地方法向量设置为0。
    d2n = depth2normal(render_pkg["depth"], mask_vis, viewpoint_cam)
    # 将渲染的深度图中计算出法线图。
    loss_depth_normal = cos_loss(render_pkg["normal"] , d2n)
    # 深度法线一致性损失。
    loss_depth = compute_depth_loss(render_pkg["depth"], curr_data['depth'], mask_vis,tracking)
    # Loss
    loss_rgb = compute_rgb_loss(render_pkg["render"], curr_data["image"], 
                                mask_vis, tracking, 
                                use_sil_for_loss=True, 
                                ignore_outlier_depth_loss=True)

    # loss_opac
    opac_ = gaussians.get_opacity
    opac_mask0 = torch.gt(opac_, 0.01) * torch.le(opac_, 0.5)
    # *是在执行逻辑与的操作。 符合条件为1 不符合条件为0
    opac_mask1 = torch.gt(opac_, 0.5) * torch.le(opac_, 0.99)
    opac_mask = opac_mask0 * 0.01 + opac_mask1
    loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()

    loss = loss_rgb + 0.1 * loss_mask +  0.01* loss_opac + loss_depth + loss_depth_normal * 0.1
    loss_dict = {"loss_rgb": loss_rgb, "loss_mask": loss_mask, "loss_opac": loss_opac, "loss_depth": loss_depth, "loss_depth_normal": loss_depth_normal}
    return loss, loss_dict, mask_vis

def prepare_output_and_logger(args):
    # 用于创建输出结果的文件夹。    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())

        args.model_path = os.path.join("./output", f"{args.source_path.split('/')[-1]}_{unique_str[0:10]}")
        
        
    # Set up output folder 写下 args 的所有参数。
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, pipe, bg, use_mask):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()[::8]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render(viewpoint, scene.gaussians, pipe, bg, [float('inf'), float('inf')])["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.get_gtImage(bg, with_mask=use_mask), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
       # 打印 lp.extract(args) 结果
    lp_params = lp.extract(args)
    print("lp.extract(args): ", lp_params.__dict__)  # 打印对象内部所有属性和值
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
