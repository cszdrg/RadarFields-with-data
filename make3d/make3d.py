import numpy as np
import open3d as o3d
import glob
import matplotlib.pyplot as plt

def visualize_radar_points(points, occ_values, threshold=0.5, seeall = True):
    """
    参数:
    points: (N, 3) 的 numpy 数组，浮点数坐标
    occ_values: (N,) 的 numpy 数组，范围 0-1 的浮点数
    threshold: 过滤阈值，低于此值的点不显示
    """
    
    # -----------------------------------------------------------
    # 1. 数据过滤 (Data Filtering)
    # -----------------------------------------------------------
    # 确保 occ_values 是 1维数组
    occ_values = occ_values.flatten()
    
    # 只保留占用率大于阈值的点
    # 这一步非常重要，能去除背景噪声，让物体轮廓更清晰
    # mask = occ_values > threshold
    mask = (occ_values > threshold) & (points[:, 2] < 1) & (points[:, 2] > -1)
    
    valid_points = points[mask]
    valid_occ = occ_values[mask]
    
    print(f"原始点数: {len(points)}, 过滤后点数: {len(valid_points)}")
    
    if len(valid_points) == 0:
        print("没有点满足阈值条件，无法绘图。")
        return

    # -----------------------------------------------------------
    # 2. 颜色映射 (Color Mapping) - 核心步骤
    # -----------------------------------------------------------
    # 使用 Matplotlib 的 colormap 将数值 (0-1) 转为 RGB 颜色
    # 推荐色系: 'jet' (经典彩虹), 'magma' (黑-红-亮黄), 'plasma' (紫-红-黄)
    # RadarFields 论文中常用类似 'plasma' 或 'inferno' 这种高对比度的热力图
    cmap = plt.get_cmap("jet") 
    
    # 归一化：确保数值严格在 0-1 之间 (以防万一)
    # norm_occ = (valid_occ - valid_occ.min()) / (valid_occ.max() - valid_occ.min() + 1e-6)
    
    # 生成颜色 (N, 3)，取 RGB，丢弃 Alpha
    colors = cmap(valid_occ)[:, :3]

    # -----------------------------------------------------------
    # 3. 构建 Open3D 点云对象
    # -----------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # T = np.array([
    #     # X_新 = 0*X_原 + (-1)*Y_原 + 0*Z_原 + 0
    #     [ 0, -1, 0, 0 ],  
    #     # Y_新 = 0*X_原 + 0*Y_原 + 1*Z_原 + 0
    #     [ 0, 0, 1, 0 ],   
    #     # Z_新 = 1*X_原 + 0*Y_原 + 0*Z_原 + 0
    #     [ 1, 0, 0, 0 ],   
    #     [ 0, 0, 0, 1 ]
    # ])
    # pcd_transformed = pcd.transform(T)
    # -----------------------------------------------------------
    # 4. 渲染设置 (Render Settings)
    # -----------------------------------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Radar Occupancy Visualization', width=1024, height=768)
    
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    T_w_c = np.eye(4)
    center = pcd.get_center()
    print(f"点云中心: {center}")
    # T_w_c[0:3, 3] = center + np.array([-10.0, 0.0, 5.0]) # 示例平移
    T_c_w = np.linalg.inv(T_w_c)

    parameters = ctr.convert_to_pinhole_camera_parameters()
    parameters.extrinsic = T_c_w
    ctr.convert_from_pinhole_camera_parameters(parameters)
    if seeall == True:
        vis.reset_view_point(True)
    
    # 获取渲染选项进行微调
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.05, 0.05, 0.05]) # 深灰色/黑色背景，显色更好
    opt.point_size = 1  # 把点稍微调大一点，形成连续的视觉感，而不是稀疏的沙砾感
    
    # 视角控制提示
    print("窗口已打开。请使用鼠标左键旋转，右键平移，滚轮缩放。")

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    
    vis.run()
    vis.destroy_window()

# ==========================================
# 模拟数据生成 (请用你的真实数据替换这里)
# ==========================================
if __name__ == "__main__":
    npy_files = sorted(glob.glob("workspace/imgs/radarfields/alpha_results/*.npy"))
    alpha = []
    points = []

    for f in npy_files:
        frame = np.load(f)
        alpha.append(frame)
    npy_files = sorted(glob.glob("workspace/imgs/radarfields/point_results/*.npy"))
    for f in npy_files:
        frame = np.load(f)
        points.append(frame)

    points = np.array(points).reshape(-1,3)

    alpha = np.array(alpha).reshape(-1)
    # print("points: ", points)
    # print("alpha: ", alpha)
    
    # 调用函数
    visualize_radar_points(points, alpha, threshold=0.99, seeall=True)