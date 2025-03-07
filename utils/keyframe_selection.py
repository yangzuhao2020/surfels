import torch
import numpy as np


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    """ 相机坐标下的三维点,并且这些点是不重复的。"""
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]] # 筛深度值

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1) 
    # 相机坐标下的三维点。
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    # idx 原始点对应的去重后索引 
    # counts 每个点出现次数
    mask = torch.isin(idx, torch.where(counts.gt(1))[0]) # 找到所有重复的点
    invalid_pt_idx = mask[:len(A)] # 去掉最后一个全0的点
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts 

def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
        从真实照片中采样1600个点,然后投影到所有关键帧上,计算每个关键帧中有多少点在图像内,然后按照这个比例排序,选择前k个关键帧。由于变换误差,或者由于当前帧和关键帧的相机位置差异,某些 3D 点在关键帧的坐标系下可能会
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where((gt_depth[0] > 0) & (gt_depth[0] < 1e10)) # Remove invalid depth values
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    sampled_indices = valid_depth_indices[indices] # 随机采样1600个点,有重复

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices) # 相机坐标下的三维点,并且这些点是不重复的。

    list_keyframe = []
    for keyframe_id, keyframe in enumerate(keyframe_list):
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1) # (N, 3) 注意这里没有归一化。
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z # 归一化
        projected_pts = points_2d[:, :2] # 去前面两列,作为投影点
        # Filter out the points that are outside the image
        edge = 20
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)
        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframe_id, 'percent_inside': percent_inside})

    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    # Select the keyframes with percentage of points inside the image > 0
    selected_keyframe_list = [keyframe_dict['id']
                                for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
    selected_keyframe_list = list(np.random.permutation(
        np.array(selected_keyframe_list))[:k])

    return selected_keyframe_list
    
    
def keyframe_selection_distance(time_idx, curr_position, keyframe_list, distance_current_frame_prob, n_samples):
    """
    Performs sampling based on a probability distribution and returns 
    the indices of `n_samples` selected keyframes.

    Args:
        probabilities: A list of probabilities representing the 
            likelihood of selecting each keyframe.
        n_samples: The number of keyframes to be sampled.

    Returns:
        A list of `n_samples` indices of the selected keyframes.
    """
    distances = []
    time_laps = []
    curr_shift = np.linalg.norm(curr_position)
    for keyframe in keyframe_list:
        est_w2c = keyframe['est_w2c'].detach().cpu()
        camera_position = est_w2c[:3, 3]
        distance = np.linalg.norm(camera_position - curr_position)
        time_lap = time_idx - keyframe['id']
        distances.append(distance)
        time_laps.append(time_lap)

    dis2prob = lambda x, scaler: np.log2(1 + scaler/(x+scaler/5))
    dis_prob = [dis2prob(d, curr_shift)+dis2prob(t, time_idx) for d, t in zip(distances, time_laps)]
    sum_prob = sum(dis_prob) / (1-distance_current_frame_prob) # distance_current_frame_prob： p_c
    norm_dis_prob = [p/sum_prob for p in dis_prob]
    norm_dis_prob.append(distance_current_frame_prob) # index 'len(keyframe_list)' indicate the current frame
    # Compute the cumulative distribution function (CDF).
    cdf = np.cumsum(norm_dis_prob)
    # Generate random samples.
    samples = np.random.rand(n_samples)
    # Select indices by comparing random numbers with CDF.
    sample_indices = np.searchsorted(cdf, samples)
    
    # no sampling ablation
    # sample_indices = []
    # for i in range(n_samples):
    #     # sample = np.random.randint(0, len(keyframe_list))
    #     sample_indices.append(len(keyframe_list))
        
    return sample_indices
