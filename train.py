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
from utils.loss_utils import l1_loss, ssim, cos_loss, bce_loss, knn_smooth_loss
from gaussian_renderer import render, network_gui
import numpy as np
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, match_depth, normal2curv, resize_image, cross_sample
from torchvision.utils import save_image
from argparse import ArgumentParser, Namespace
import time
import os
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, opt.camera_lr, shuffle=False, resolution_scales=[1, 2, 4])
    use_mask = dataset.use_mask
    gaussians.training_setup(opt) # 对模型参数进行初始化、优化器设置以及学习率调度器配置。
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif use_mask: # visual hull init
        gaussians.mask_prune(scene.getTrainCameras(), 4) # 掩码剪枝
        None

    opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))
    # 调整密度的周期。
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    background = torch.rand((3), dtype=torch.float32, device="cuda") if dataset.random_background else background
    patch_size = [float('inf'), float('inf')] # 设置两个无穷大的元素。
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    pool = torch.nn.MaxPool2d(9, stride=1, padding=4)
    # 在不改变特征图尺寸的情况下，提取局部区域的最大值

    viewpoint_stack = None # 记录相机视角
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    count = -1
    for iteration in range(first_iter, opt.iterations + 2):

        iter_start.record()

        gaussians.update_learning_rate(iteration) # 传入迭代次数，修改位置参数的学习率。
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration - 1 == 0:
            scale = 4
        elif iteration - 1 == 2000 + 1:
            scale = 2
        elif iteration - 1 == 5000 + 1:
            scale = 1
        # scale = 1
        # 在不同的训练轮次中会启用不同的分辨率，以至于产生了一个缩放比例 scale.

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale).copy()[:]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        # 打印 viewpoint_cam 的类型
        # print("===== gaussians.config =====")
        # print(type(gaussians.config))

        # # 打印 viewpoint_cam 的内容
        # print("===== gaussians.config =====")
        # print(gaussians.config)
        # viewpoint_cam = scene.getTrainCameras(scale)[0]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, patch_size)
        render_image, render_normal, render_depth, render_opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        print("===== gaussians.config =====")
        print(render_depth.shape)
        mask_gt = viewpoint_cam.get_gtMask(use_mask)
        gt_image = viewpoint_cam.get_gtImage(background, use_mask)
        mask_vis = (render_opac.detach() > 1e-5)
        # 确定哪些像素是背景，要求把不透明度很低的点找出来，那么它对应的区域说明是不存在物体。
        loss_mask = (render_opac * (1 - pool(mask_gt))).mean()
        # 为什么会出现池化这个操作，原因很简单，他想要扩大边界一点。
        
        render_normal = torch.nn.functional.normalize(render_normal, dim=0) * mask_vis
        # 归一化，然后将区域对应背景的地方法向量设置为0。
        d2n = depth2normal(render_depth, mask_vis, viewpoint_cam)
        # 将渲染的深度图中计算出法线图。
        loss_depth_normal = cos_loss(render_normal, d2n)
        loss_depth_normal = (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_depth_normal
        # 深度法线一致性损失。

        mono = viewpoint_cam.mono if dataset.mono_normal else None
        # 已经预处理好的单目法线图。
        if mono is not None:
            mono *= mask_gt
            monoN = mono[:3]
            # monoD = mono[3:]
            # monoD_match, mask_match = match_depth(monoD, depth, mask_gt * mask_vis, 256, [viewpoint_cam.image_height, viewpoint_cam.image_width])
            loss_monoN = cos_loss(render_normal, monoN, weight=mask_gt)
            loss_monoN = (0.04 - ((iteration / opt.iterations)) * 0.02) * loss_monoN 
            loss_monoN = loss_monoN + l1_loss(normal2curv(render_normal, mask_vis),0) * 0.005

        # Loss
        Ll1 = l1_loss(render_image, gt_image)
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(render_image, gt_image))

        # loss_opac
        opac_ = gaussians.get_opacity
        opac_mask0 = torch.gt(opac_, 0.01) * torch.le(opac_, 0.5)
        # *是在执行逻辑与的操作。 符合条件为1 不符合条件为0
        opac_mask1 = torch.gt(opac_, 0.5) * torch.le(opac_, 0.99)
        opac_mask = opac_mask0 * 0.01 + opac_mask1
        loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()

        loss = loss_rgb + 0.1 * loss_mask +  0.01* loss_opac + loss_monoN + loss_depth_normal

        loss.backward() # 反向传播。

        iter_end.record()
        

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_rgb.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Pts={len(gaussians._xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            test_background = background
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, pipe, test_background, use_mask)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration > opt.densify_from_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 记录最大的投影半径。
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                min_opac = 0.1
                if iteration % opt.densification_interval == 0:
                    gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                    gaussians.adaptive_densify(opt.densify_grad_threshold, # 密度增强的梯度阈值。 这里可能会对误差大的地方加点。
                                               scene.cameras_extent) 
                
                if (iteration - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0:
                    gaussians.reset_opacity(0.12, iteration)
                # 定期会对透明度进行重置。
                """
                高斯体素密度调整机制
                记录最大投影尺寸：判断哪些点始终“不可见”或“无贡献”。
                统计高斯点分布数据：辅助自适应密度增强和剪枝策略。
                剪枝 (Prune)：删除贡献小或透明度过低的高斯点，减少计算成本。
                扩增 (Densify)：在高误差区域补充新的高斯点，提升渲染质量。
                定期重置不透明度：给所有高斯点“复活”机会，防止有用点被过早剪除。
                """

            if (iteration - 1) % 1000 == 0:
                normal_wrt = normal2rgb(render_normal, mask_vis)
                depth_wrt = depth2rgb(render_depth, mask_vis)
                img_wrt = torch.cat([gt_image, render_image, normal_wrt * render_opac, depth_wrt * render_opac], 2)
                # 图片拼接
                save_image(img_wrt.cpu(), f'test/test.png')
                

            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()
                # viewpoint_cam.optimizer.step()
                # viewpoint_cam.optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                # gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


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
