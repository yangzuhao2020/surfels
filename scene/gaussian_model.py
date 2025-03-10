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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, normal2rotation, compute_normals_cross_product
from torch import nn
import os
from torch.utils.cpp_extension import load
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
# from scene.colmap_loader import qvec2rotmat
from utils.general_utils import quaternion2rotmatrix
# from utils.graphics_utils import BasicPointCloud
from utils.image_utils import world2scrn
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.image_utils import energy_mask


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args):
        self.active_sh_degree = 0
        self.max_sh_degree = 0 
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.scale_gradient_accum = torch.empty(0)
        self.rot_gradient_accum = torch.empty(0)
        self.opac_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        try:
            self.config = [args.surface, args.normalize_depth, args.perpix_depth]
        except AttributeError:
            self.config = [True, True, True]
        self.setup_functions()
        self.utils_mod = load(name="cuda_utils", sources=["utils/ext.cpp", "utils/cuda_utils.cu"])
        self.opac_reset_record = [0, 0]

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.scale_gradient_accum,
            self.rot_gradient_accum,
            self.opac_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.config
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        scale_gradient_accum,
        rot_gradient_accum,
        opac_gradient_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.config) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.scale_gradient_accum = scale_gradient_accum
        self.rot_gradient_accum = rot_gradient_accum
        self.opac_gradient_accum = opac_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # print(self._scaling)
        return self.scaling_activation(self._scaling)
        # scaling_2d = torch.cat([self._scaling[..., :2], torch.full_like(self._scaling[..., 2:], -1e10)], -1)
        # return self.scaling_activation(scaling_2d)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    @property
    def get_normal(self):
        return quaternion2rotmatrix(self.get_rotation)[..., 2]


    # def oneupSHdegree(self):
    #     if self.active_sh_degree < self.max_sh_degree:
    #         self.active_sh_degree += 1
    
    def create_pcd(self, color, depth, k, w2c, mask=None, add = True):
        """ color 的形状 (C,H,W) depth (1,H,W) mask (H,W) """   
    # def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # 从点云中初始化，设置好初始的各个参数情况。设置梯度追踪。
        # self.spatial_lr_scale = spatial_lr_scale

        # fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0], "Shape of points at initialisation : ",fused_point_cloud.shape)

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # # 将点云坐标转换为 PyTorch 张量 限制输出的值最小为 0.0000001 返回 每个点的 KNN 平均距离。
        # scales = torch.log(torch.sqrt(dist2 / 4))[...,None].repeat(1, 3)
        # # scales 是一个 形状为 [N, 3] 的张量, 用于缩放点云的 x, y, z 坐标, 也同时可以表达密度。
        # # scales = torch.log(torch.ones((len(fused_point_cloud), 3)).cuda() * 0.02)
        
        # if self.config[0] > 0:
        #     if np.abs(np.sum(pcd.normals)) < 1: # 判断是否绝对值是否小于 1。
        #         dup = 4
        #         fused_point_cloud = torch.cat([fused_point_cloud for _ in range(dup)], 0)
        #         # 复制点云数据 4 份
        #         fused_color = torch.cat([fused_color for _ in range(dup)], 0)
        #         # 复制颜色 4 份
        #         scales = torch.cat([scales for _ in range(dup)], 0)
        #         # 复制尺度因子 4 份
        #         normals = np.random.rand(len(fused_point_cloud), 3) - 0.5
        #         # 随机生成法向量。范围在 [-0.5,0.5]
        #         normals /= np.linalg.norm(normals, 2, 1, True)
        #         # 归一化。
        #     else:
        #         normals = pcd.normals

        #     rots = normal2rotation(torch.from_numpy(normals).to(torch.float32)).to("cuda")
        #     scales[..., -1] -= 1e10 # squeeze z scaling
            # 抑制Z方向的缩放。
            
        #     # scales[..., -1] = 0
        #     # print(pcd.normals)
        #     # exit()
        #     # rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        #     # rots = self.rotation_activation(rots)
        # else:
        #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        #     rots[:, 0] = 1
        
        width, height = color.shape[2], color.shape[1]
        CX = k[0][2]  # 主点 x 坐标
        CY = k[1][2]  # 主点 y 坐标
        FX = k[0][0]  # 焦距 x
        FY = k[1][1]  # 焦距 y

        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth[0].reshape(-1)

        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)  # (X, Y, Z, 1)
        c2w = torch.inverse(w2c)  # 计算世界坐标变换矩阵
        pts = (c2w @ pts4.T).T[:, :3].to("cuda")  # 从相机坐标变换到世界坐标 (H * W, 3)

        scale_gs = depth_z / ((FX + FY)/2)
        scales = scale_gs.square().unsqueeze(-1).repeat(1, 3).to("cuda")  # (H * W, 3)
        col = color.transpose(0, 1, 2).reshape(-1, 3).to("cuda")
        
        # Select points based on mask
        if mask is not None:
            mask = mask.reshape(-1).bool().to("cuda")  # 确保 mask 形状正确
            col = col[mask]
            pts = pts[mask]
            scales = scales[mask]
            
        normals = compute_normals_cross_product(pts, width, height)
        rots = normal2rotation(torch.from_numpy(normals).to(torch.float32)).to("cuda")
        scales[..., -1] -= 1e10 # squeeze z scaling
        # features = torch.zeros((pts.shape[0], 3)).float().cuda()
        # features 张量大小 (n,3,(max_sh_degree + 1)^2) 3表示RGB 用于颜色表示。
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0
        # 球谐0阶系数，其值设置为RGB的颜色，其他球谐分量设置为0。
        opacities = inverse_sigmoid(0.1 * torch.ones((pts.shape[0], 1), dtype=torch.float, device="cuda"))
        # 初始化透明度。
        if add:
            self._xyz = nn.Parameter(pts.requires_grad_(True))
            self._features_dc = nn.Parameter(col.requires_grad_(True))
            # 存储球谐0阶，同时改变维度顺序。(N, 3)
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            # 每个点的最大投影半径（2D 视角下），初始化为 0。
            # exit()
        else:
            return pts, col, scales, rots, opacities

    def add_new_gaussians(self, curr_data, render_pkg, viewpoint_cam):
        # 确定太远的点
        depth_error = torch.abs(curr_data["gt_depth"] - render_pkg["depth"])* (curr_data["gt_depth"] > 0)
        non_presence_depth_mask = (render_pkg["depth"] > curr_data["gt_depth"]) * (depth_error > 20 * depth_error.mean())
        non_presence_depth_mask = non_presence_depth_mask.reshape(-1)
        
        if torch.sum(non_presence_depth_mask) > 0:
            curr_cam_rot = torch.nn.functional.normalize(viewpoint_cam.q.detach())
            curr_cam_tran = viewpoint_cam.T.detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran
            mask = (curr_data["depth"] > 0) & energy_mask(curr_data["color"])
            mask = non_presence_depth_mask & mask
            pts, col, scales, rots, opacities = self.create_pcd(curr_data["color"], curr_data["depth"], curr_data["k"],curr_w2c, mask, add=False)
            self.densification_postfix(pts, col, scales, rots, opacities)
        
    def training_setup(self, training_args, tracking=False):
        # 对模型梯度累积进行初始化、优化器设置以及学习率调度器配置。
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 位置参数的梯度累积。
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 缩放参数的梯度累积。
        self.rot_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 旋转参数的梯度累积。
        self.opac_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 不透明度参数的梯度累积。
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        track = 1
        if tracking:
            track = 0
            
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * track, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * track, "name": "f_dc"}, # 第0阶球谐矩阵的颜色
            {'params': [self._features_rest], 'lr': 0.0, "name": "f_rest"}, # 除第0阶球谐矩阵的颜色系数。
            {'params': [self._opacity], 'lr': training_args.opacity_lr * track, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * track, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * track, "name": "rotation"}
        ]

        self.config[3] = training_args.camera_lr > 0
        # self.optimizer = torch.optim.SGD(l)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init)
        # self.xyz_scheduler_args 这是一个函数，输入训练的实时步数，既可以得到位置的学习率。

    def update_learning_rate(self, iteration, tracking = False):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                if tracking:
                    param_group['lr'] = 0
                param_group['lr'] = lr 

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def save_pcl(self, path):
    #     v = self.get_xyz
    #     n = self.get_normal
    #     c = SH2RGB(self._features_dc)[:, 0]
    #     save_pcl('test/pcl.ply', v, n, c)  
   
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]
        self.rot_gradient_accum = self.rot_gradient_accum[valid_points_mask]
        self.opac_gradient_accum = self.opac_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {} # 用于存储更新后的参数，并在最后返回。
        for group in self.optimizer.param_groups: # 遍历优化器中的每个参数组。
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]] # 取出要追加的新张量 
            stored_state = self.optimizer.state.get(group['params'][0], None) # 得到优化器的状态。
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]] # 删除旧状态。
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)) # 拼接新的参数
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
    def densification_postfix(self, new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        # "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.rot_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, pre_mask=True):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(selected_pts_mask.dtype, pre_mask.dtype)
        # selected_pts_mask *= pre_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        if self.config[0] > 0:
            new_scaling[:, -1] = -1e10
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, pre_mask=True):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask += (grad_rot > grad_rot_thrsh).squeeze()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        selected_pts_mask *= pre_mask
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation)

    def adaptive_prune(self, min_opacity, extent):

        # print(sum(grad_rot > 1.2) / len(grad_rot))
        # print(sum(grad_pos > max_grad) / len(grad_pos), max_grad)

        n_ori = len(self._xyz)

        # prune
        # prune_mask = 
        # opac_thrsh = torch.tensor([min_opacity, 1])
        opac_temp = self.get_opacity
        prune_opac =  (opac_temp < min_opacity).squeeze()
        # prune_opac += (opac_temp > opac_thrsh[1]).squeeze()

        # scale_thrsh = torch.tensor([2e-4, 0.1]) * extent
        scale_min = self.get_scaling[:, :2].min(1).values
        scale_max = self.get_scaling[:, :2].max(1).values
        prune_scale = scale_max > 0.5 * extent
        prune_scale += (scale_min * scale_max) < (1e-8 * extent**2)
        # print(prune_scale.sum())
        
        prune_vis = (self.denom == 0).squeeze()
        prune = prune_opac + prune_vis + prune_scale
        self.prune_points(prune)
        # print(f'opac:{prune_opac.sum()}, scale:{prune_scale.sum()}, vis:{prune_vis.sum()} extend:{extent}')
        # print(f'prune: {n_ori}-->{len(self._xyz)}')

    def adaptive_densify(self, max_grad, extent):
        grad_pos = self.xyz_gradient_accum / self.denom
        grad_scale = self.scale_gradient_accum /self.denom
        grad_rot = self.rot_gradient_accum /self.denom
        grad_opac = self.opac_gradient_accum /self.denom
        grad_pos[grad_pos.isnan()] = 0.0
        grad_scale[grad_scale.isnan()] = 0.0
        grad_rot[grad_rot.isnan()] = 0.0
        grad_opac[grad_opac.isnan()] = 0.0


        # densify
        # opac_lr = [i['lr'] for i in self.optimizer.param_groups if i['name'] == 'opacity'][0]
        larger = torch.le(grad_scale, 1e-7)[:, 0] #if opac_lr == 0 else True
        # print(grad_opac.min(), grad_opac.max(), grad_opac.mean())
        denser = torch.le(grad_opac, 2)[:, 0]
        pre_mask = denser * larger
        
        self.densify_and_clone(grad_pos, max_grad, extent, pre_mask=pre_mask)
        self.densify_and_split(grad_pos, max_grad, extent)


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # print(self.xyz_gradient_accum.shape)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # print(self.xyz_gradient_accum.shape)
        # print(self._scaling.grad.shape)
        # exit()
        self.scale_gradient_accum[update_filter] += self._scaling.grad[update_filter, :2].sum(1, True)
        # print(self._scaling.grad)
        self.rot_gradient_accum[update_filter] += torch.norm(self._rotation[update_filter], dim=-1, keepdim=True)
        self.opac_gradient_accum[update_filter] += self._opacity[update_filter]
        self.denom[update_filter] += 1


    def mask_prune(self, cams, pad=4):
        batch_size = 32
        batch_num = len(cams) // batch_size + int(len(cams) % batch_size != 0)
        cams_batch = [cams[i * batch_size : min(len(cams), (i + 1) * batch_size)] for i in range(batch_num)]
        # 使用列表推导式生成了一个新列表 cams_batch，其中每一个元素都是从 cams 中提取的一个子列表。
        for c in cams_batch:
            _, _, inMask, outView = world2scrn(self._xyz.detach(), c, pad)
            visible = inMask.all(0) * ~(outView.all(0))
            if list(visible.shape) != []:
                self.prune_points(~visible)
        # 移除在各个视角中都不可见的点。

    def to_occ_grid(self, cutoff, grid_dim_max=512, bound_overwrite=None):
        if bound_overwrite is None:
            xyz_min = self._xyz.min(0)[0]
            xyz_max = self._xyz.max(0)[0]
            xyz_len = xyz_max - xyz_min
            xyz_min -= xyz_len * 0.1
            xyz_max += xyz_len * 0.1
        else:
            xyz_min, xyz_max = bound_overwrite
        xyz_len = xyz_max - xyz_min

        # print(xyz_min, xyz_max, xyz_len)
        
        # grid_dim_max = 1024
        grid_len = xyz_len.max() / grid_dim_max
        grid_dim = (xyz_len / grid_len + 0.5).to(torch.int32)

        grid = self.utils_mod.gaussian2occgrid(xyz_min, xyz_max, grid_len, grid_dim,
                                               self.get_xyz, self.get_rotation, self.get_scaling, self.get_opacity,
                                               torch.tensor([cutoff]).to(torch.float32).cuda())
        
        
        # print('here')
        # x, y, z = torch.meshgrid(torch.arange(0, grid_dim[0]), torch.arange(0, grid_dim[1]), torch.arange(0, grid_dim[2]), indexing='ij')
        
        # print('here')
        # exit()
        
        # grid_cord = torch.stack([x, y, z], -1).cuda()

        return grid, -xyz_min, 1 / grid_len, grid_dim