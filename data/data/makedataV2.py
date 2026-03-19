from scipy.spatial.transform import Rotation as R
from torchvision import transforms
import torch
from pyproj import Proj, Transformer
from pathlib import Path
import shutil
import numpy as np
import json
import os
import cv2
from PIL import Image
import glob
import pandas as pd
import tqdm
import math
from typing import Optional
import matplotlib.pyplot as plt
# ------------------------------
# 生成位姿矩阵
def euler_to_rotmat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, sy, 0],   # 原来是 [[cy, -sy, 0],
                    [-sy, cy, 0],  #          [sy, cy, 0],
                    [0, 0, 1]])   #          [0, 0, 1]])
    Ry = np.array([[cp, 0, sp],
                   [ 0, 1, 0],
                   [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])

    R = Rx @ Ry @ Rz
    return R


def load_odometry(csv_path):
    """
    读取 radar_odometry.csv 文件。
    假设包含字段：
      source_radar_timestamp, destination_radar_timestamp, x, y, z, roll, pitch, yaw
    """
    df = pd.read_csv(csv_path)
    odom = {}

    for _, row in df.iterrows():
        src, dst = int(row['source_radar_timestamp']), int(row['destination_radar_timestamp'])
        t = np.array([row['x'], row['y'], row['z']])
        R = euler_to_rotmat(row['roll'], row['pitch'], row['yaw'])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        # odom[(dst, src)] =  np.linalg.inv(T)
        odom[(dst, src)] =  T
    return odom


def get_pose_from_start(odom_dict, start_ts, index_ts):
    """
    计算 index_ts 相对 start_ts 的位姿矩阵。
    odom_dict 存储相邻帧间的相对变换。
    """
    # 提取所有出现过的时间戳
    timestamps = sorted(set([s for s, _ in odom_dict.keys()] + [d for _, d in odom_dict.keys()]))

    if start_ts not in timestamps or index_ts not in timestamps:
        raise ValueError("timestamp not found in odometry data")

    start_idx = timestamps.index(start_ts)
    index_idx = timestamps.index(index_ts)
    # print("计算当前位姿时 最终位姿的索引：", index_idx)
    T = np.eye(4)
    for i in range(start_idx, index_idx):
        src, dst = timestamps[i], timestamps[i+1]
        T = T @ odom_dict[(src, dst)]

    return T, T[:3, 3][0], T[:3, 3][1], T[:3, 3][2]
# ------------------------------
# 生成占用信息
def computational_occupancy_estimator(
    Pr_raw: np.ndarray,
    prior_occupancy: Optional[np.ndarray] = None,
    delta: float = 10,
    Po: float = 0.1,
    occlusion_decay_rate: float = 0.044,
    strong_reflection_threshold: float = 0.001
) -> np.ndarray:
    """
    计算型占用率估算器 (Computational Occupancy Estimator) 的实现。

    参数:
        Pr_raw (np.ndarray): 原始 FFT 雷达功率矩阵，形状为 (距离单元数, 方位角数)。
        prior_occupancy (Optional[np.ndarray]): 先前的占用率估计 P_c。如果为 None，则使用 0.5 的中性先验。
        delta (float): 用于初始概率估计公式中的超参数 δ。
        Po (float): 用于初始概率估计公式中的功率阈值 P_o (应为归一化后的值)。
        occlusion_decay_rate (float): 遮挡模型中的指数衰减率 λ。
        strong_reflection_threshold (float): 定义“强反射点”的归一化功率阈值。

    返回:
        np.ndarray: 当前帧的最终占用率估计 O(Pr)。
    """
    
    R, A = Pr_raw.shape # R: 距离单元数 (Range), A: 方位角数 (Azimuth)
    
    # ----------------------------------------------------
    # --- 步骤 1: 动态阈值滤波 (Dynamic Thresholding) ---
    # ----------------------------------------------------
    
    # 1.1 独立噪声估计
    # 估算每个方位角（列）的噪声阈值
    n_phi = np.median(Pr_raw, axis=0) # shape (A,)
    # 估算每个距离单元（行）的噪声阈值
    n_b = np.median(Pr_raw, axis=1) # shape (R,)
    
    # 1.2 扩展矩阵并计算最终阈值 T(phi, b)
    # 将一维中位数广播到 (R, A) 形状
    N_phi = np.tile(n_phi[np.newaxis, :], (R, 1))
    N_b = np.tile(n_b[:, np.newaxis], (1, A))
    
    # T(phi, b) = 2 * max(n_phi, n_b)
    T = 2 * np.maximum(N_phi, N_b)
    
    # 1.3 滤波
    # Pr_prime 是保留的功率值 (Pr >= T)
    Pr_prime = np.where(Pr_raw >= T, Pr_raw, 0.0)
    
    # 1.4 归一化 (为下一步做准备)
    max_Pr_prime = np.max(Pr_prime)
    if max_Pr_prime > 0:
        # 将保留的功率值归一化到 [0, 1] 范围
        Pr_norm_prime = Pr_prime / max_Pr_prime
    else:
        Pr_norm_prime = Pr_prime # 保持全零
        
    # -----------------------------------------------------------------
    # --- 步骤 2: 初始占用概率估计 (Initial Occupancy Probability) ---
    # -----------------------------------------------------------------
    
    # 公式: p_r(phi, b) = Pr'(phi, b) * exp(delta * (Pr'(phi, b) - Po))
    
    # 计算指数项
    exp_term = np.exp(delta * (Pr_norm_prime - Po))
    
    # 初始概率估计 p_r
    pr_initial = Pr_norm_prime * exp_term
    
    # 确保结果在概率范围 [0, 1] 内
    pr_initial = np.clip(pr_initial, 0.0, 1.0)
    
    # -------------------------------------------------------
    # --- 步骤 3: 遮挡模型集成 (Occlusion Model Integration) ---
    # -------------------------------------------------------
    
    pr_occlusion = pr_initial.copy()
    
    # 遮挡模型需要迭代每个方位角 (列)
    for a in range(A):
        # 找到当前方位角上所有的强反射点 (即非零或大于阈值的归一化功率)
        strong_reflections = np.where(Pr_norm_prime[:, a] > strong_reflection_threshold)[0]
        
        # 遍历当前方位角上的每个距离单元 r
        for r in range(R):
            # 找到最近的、距离小于 r 的强反射点 r_p (b_p < b)
            bp_indices = strong_reflections[strong_reflections < r]
            
            if len(bp_indices) > 0:
                # r_p 即为 b_p：最近的强反射点索引
                r_p = bp_indices[-1] 
                
                # 距离差 dx = b_p - b。由于 b_p < b, dx 为负值，表示衰减
                dx = r_p - r
                
                # 计算衰减后的概率：Pr(phi, b_p) * exp(lambda * dx)
                attenuated_prob = Pr_norm_prime[r_p, a] * np.exp(dx / occlusion_decay_rate)

                # 更新概率：取初始概率 pr_initial 和衰减概率中的最大值
                pr_occlusion[r, a] = np.maximum(pr_initial[r, a], attenuated_prob)
                
    # ------------------------------------------
    # --- 步骤 4: 贝叶斯更新 (Bayesian Update) ---
    # ------------------------------------------
    
    # 设置先验占用率 P_prior
    if prior_occupancy is None:
        # 如果未提供先验，使用中性先验 0.5
        P_prior = 0.5 * np.ones((R, A))
    else:
        P_prior = prior_occupancy
        
    # p_new 是经过遮挡模型处理后的最新概率估计
    p_new = pr_occlusion
    
    # 贝叶斯更新公式:
    # O_new = (p_new * P_prior) / (p_new * P_prior + (1 - p_new) * (1 - P_prior))
    
    numerator = p_new * P_prior
    denominator = numerator + (1 - p_new) * (1 - P_prior)
    
    # 使用 np.divide 安全地执行除法，避免除以零
    O_new = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    return O_new
# --------------------------------
# 可视化位姿
def plot_trajectory(poses, path, size=0.05):
    """## Save 3D plot of sensor trajectory from GNSS poses"""
    num_poses = poses.shape[0]

    radar_center = torch.tensor([0.0, 0.0, 0.0], device=poses.device)
    radar_forward = torch.tensor([1.0, 0.0, 0.0], device=poses.device)
    radar_forward = radar_forward[None,:,None].expand((num_poses, 3, 1))

    # Mapping center and forward vector to world coords
    # for each frame using corresponding pose
    world_centers = radar_center + poses[:,:3,-1] # [num_poses, 3]
    world_forwards = torch.bmm(poses[:,:3,:3], radar_forward)[...,0] # [num_poses, 3]

    world_centers = world_centers.detach().cpu().numpy()
    world_forwards = world_forwards.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=0.0, azim=320.0)
    ax.view_init(elev=30, azim=320)
    ax.dist = 6.0
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(world_centers[...,0], world_centers[...,1], world_centers[...,2], s=size*5,  c='blue')
    for p, f in zip(world_centers, world_forwards):
        ax.plot([p[0], p[0]+f[0]],
                [p[1], p[1]+f[1]],
                [p[2], p[2]+f[2]],
                color='red')
    # ax.scatter(world_centers[...,0], world_centers[...,1], world_centers[...,2], s=size*5)
    # ax.quiver(world_centers[...,0], world_centers[...,1], world_centers[...,2],
    #           world_forwards[...,0], world_forwards[...,1], world_forwards[...,2],
    #           length=size*13)
    fig.canvas.draw()
    fig.savefig(path, dpi=300)
    plt.close()
# ------------------------------
# 删除文件
folder = Path("preprocess_results\\occupancy_component\\preprocess_results")
for item in folder.iterdir():
    if item.is_file():
        item.unlink()              # 删除文件
    elif item.is_dir():
        shutil.rmtree(item)        # 删除子目录及其内容
folder = Path("preprocess_results\\thresholded_fft\\seq10")
for item in folder.iterdir():
    if item.is_file():
        item.unlink()              # 删除文件
    elif item.is_dir():
        shutil.rmtree(item)        # 删除子目录及其内容
folder = Path("data\\seq10\\radar")
for item in folder.iterdir():
    if item.is_file():
        item.unlink()              # 删除文件
    elif item.is_dir():
        shutil.rmtree(item)        # 删除子目录及其内容 
folder = Path("checkpoints")
for item in folder.iterdir():
    if item.is_file():
        item.unlink()              # 删除文件
    elif item.is_dir():
        shutil.rmtree(item)        # 删除子目录及其内容 
folder = Path("workspace")
for item in folder.iterdir():
    if item.is_file():
        item.unlink()              # 删除文件
    elif item.is_dir():
        shutil.rmtree(item)        # 删除子目录及其内容 

odom_dict = load_odometry("data/data/radar_odometry.csv")

frames = []
frame_id = 0
i = 0
png_files = sorted(f for f in os.listdir('data/data/radar') if f.lower().endswith(".png"))
first = '1547121830878312.png' # 从雷达数据集first

f = False
for file in png_files:
    if str(file) != first and f == False:
        continue
    f = True
    i += 1
    if i > 80:
        break
    if file.lower().endswith(".png"):
        timestamp = file
        frames.append((frame_id, timestamp))
    frame_id += 1



train_indices = []
test_indices = []
radar2worlds = []
timestamps_radar = []

# 根据帧名获取位姿矩阵 生成数据
FFTPath = "data\\data\\radar"

xx = []
yy = []
zz = []
start_ts = int(frames[0][1].split('.')[0])
all_FFT_Data_Sim =  np.array([])
for frame_id, timestamp in tqdm.tqdm(frames):
    # 计算位姿
    T, x, y, z = get_pose_from_start(odom_dict, start_ts, int(timestamp.split('.')[0]))
    xx.append(x)
    yy.append(y)
    zz.append(z)
    radar2worlds.append(T.tolist())
    timestamps_radar.append(timestamp)

    # filename = f"{frame_id:06d}.png"
    filename = timestamp
    old_path = os.path.join(FFTPath, filename)
    new_filename = timestamp
    new_path = os.path.join("data\\seq10\\radar", new_filename)
    shutil.copy2(old_path, new_path)

    FFT_Data_Sim = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    FFT_Data_Sim = FFT_Data_Sim[:, 11:1500]
    if all_FFT_Data_Sim.size == 0:
        all_FFT_Data_Sim = FFT_Data_Sim
    else:
        all_FFT_Data_Sim = np.concatenate([all_FFT_Data_Sim, FFT_Data_Sim], axis = 0)

    # 生成占用信息
    occ = computational_occupancy_estimator(FFT_Data_Sim.T).T
    new_filename = new_filename.split('.')[0] + '.npy'
    np.save(os.path.join('preprocess_results/occupancy_component/preprocess_results', new_filename), occ)


# 阈值化处理
maxxFFT = all_FFT_Data_Sim.max()
print("maxx FFT = ", maxxFFT)
# thre = np.median(all_FFT_Data_Sim)
thre = 0.15
paths = sorted(glob.glob(os.path.join("data\\seq10\\radar\\*.png")))
for t, path in enumerate(paths):
    base = os.path.splitext(os.path.basename(path))[0]
    FFT_Data_Sim = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    FFT_Data_Sim = FFT_Data_Sim[:, 11:1500] / maxxFFT
    FFT_Data_Sim[FFT_Data_Sim < thre] = 0
    # FFT_Data_Sim = FFT_Data_Sim  / maxxFFT
    np.save(os.path.join('preprocess_results\\thresholded_fft\\seq10', f"{base}.npy"), FFT_Data_Sim)

all_indices = np.arange(0, 80)  # 0~80 共 80 个
np.random.shuffle(all_indices)  # 打乱顺序
train_indices = all_indices[:60].tolist()  # 前 60 个作为训练
test_indices  = all_indices[60:80].tolist()  # 后 20 个作为测试
offsets = [- (np.max(xx) + np.min(xx)) / 2, - (np.max(yy) + np.min(yy)) / 2, - (np.max(zz) + np.min(zz)) / 2]
# offsets = [- np.min(xx), - np.min(yy), - np.min(zz)]
# scalers = [150, 150, 150]
scalers = [np.max(xx) - np.min(xx) + 100, np.max(yy) - np.min(yy) + 100,  np.max(zz) - np.min(zz) + 100]
print("offset = ", offsets)
print("scalers = ", scalers)
print("maxx x = ", np.max(xx))
print("minn x= ", np.min(xx))
print("maxx y= ", np.max(yy))
print("minn y= ", np.min(yy))
print("maxx z= ", np.max(zz))
print("minn z= ", np.min(zz))
print(np.max(xx) - np.min(xx)) 
print(np.max(yy) - np.min(yy))
print(np.max(zz) - np.min(zz))

data = {
    "test_indices": test_indices,
    "train_indices": train_indices,
    "radar2worlds": radar2worlds,
    "timestamps_radar": timestamps_radar,
    "offsets": offsets,
    "scalers": scalers
}
plot_trajectory(torch.tensor(data['radar2worlds']), "data/data/pose", size=0.05)
with open('preprocess_results\\preprocess_results.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("done")

