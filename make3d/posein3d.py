import numpy as np
import open3d as o3d
import imageio
import matplotlib.pyplot as plt
import glob
import json

def create_dummy_point_cloud(points, occ_values, threshold=0.5):
    # -----------------------------------------------------------
    # 1. 数据过滤 (Data Filtering)
    # -----------------------------------------------------------
    
    # 只保留占用率大于阈值的点
    # 这一步非常重要，能去除背景噪声，让物体轮廓更清晰
    mask = occ_values > threshold
    
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
    occ_norm = (valid_occ - valid_occ.min()) / (valid_occ.max() - valid_occ.min() + 1e-6)
    occ_norm = np.clip((occ_norm ** 2.0), 0, 1)
    cmap = plt.get_cmap("terrain") 

    # 生成颜色 (N, 3)，取 RGB，丢弃 Alpha
    colors = cmap(occ_norm)[:, :3]

    # -----------------------------------------------------------
    # 3. 构建 Open3D 点云对象
    # -----------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # R_conv = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    # pcd.rotate(R_conv, center=(0, 0, 0))
    return pcd

# ==========================================
# 核心功能：保存 GIF
# ==========================================
def save_trajectory_gif(pcd, poses, output_filename="output.gif", fps=15):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Recording...", width=640, height=480, visible=True)

    vis.add_geometry(pcd)

    width, height = 300, 300  # 根据点云范围调节
    mesh_ground = o3d.geometry.TriangleMesh.create_box(width=width, height=5, depth=height)
    vertices = np.asarray(mesh_ground.vertices)
    colors = np.zeros_like(vertices)
    colors[:, 0] = 0 + 0.1 * (vertices[:, 0]/width)  # R
    colors[:, 1] = 0 + 0.1 * (vertices[:, 2]/height)  # G
    colors[:, 2] = 0                                # B
    mesh_ground.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_ground.translate([-width/2, 0, -height/2])  # 放置到 y=0
    vis.add_geometry(mesh_ground)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.asarray([0.7, 0.7, 0.7])
    opt.point_size = 5
    # opt.light_on = True  # 开启光照

    ctr = vis.get_view_control()
    parameters = ctr.convert_to_pinhole_camera_parameters()

    frames_buffer = []

    for i, pose in enumerate(poses):

        parameters.extrinsic = np.linalg.inv(pose)

        ctr.convert_from_pinhole_camera_parameters(parameters)

        # vis.run()
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(do_render=True)
        img_uint8 = (np.asarray(img) * 255).astype(np.uint8)
        frames_buffer.append(img_uint8)

        print(f"\rProcessing frame: {i+1}/{len(poses)}", end="")

    imageio.mimsave(output_filename, frames_buffer, fps=fps, loop=0)
    vis.destroy_window()

# ==========================================
# 运行
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

    T3 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float32)

    points = points @ T3.T

    points[:, 1] *= -1
    alpha = np.array(alpha).reshape(-1)
    
    point_cloud = create_dummy_point_cloud(points, alpha, threshold=0.25)
    poses_radar = []

    with open("preprocess_results/preprocess_results.json") as f:
            preprocess = json.load(f)
    for f in preprocess["radar2worlds"]:
            pose_radar = np.array(f, dtype=np.float32) # [4, 4]
            poses_radar.append(pose_radar)
    T = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    camera_poses = T @ np.array(poses_radar) @ np.linalg.inv(T)

    save_trajectory_gif(point_cloud, camera_poses, "my_scan_result.gif")