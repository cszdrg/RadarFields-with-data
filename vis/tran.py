import os
import glob
import numpy as np
import cv2
from pathlib import Path
import shutil

# ============================================================
# 极坐标转笛卡尔坐标
# ============================================================
def radar_polar_to_cartesian(fft_data: np.ndarray,
                             radar_resolution: float,
                             cart_resolution: float,
                             cart_pixel_width: int,
                             interpolate_crossover=True) -> np.ndarray:
    """
    fft_data: shape = [num_azimuths, num_ranges]
    radar_resolution: 每个 range bin 的距离分辨率 (米)
    """

    num_azimuths = fft_data.shape[0]
    azimuths = np.linspace(0, 2 * np.pi, num_azimuths, endpoint=False, dtype=np.float32)

    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution

    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -coords)

    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    azimuth_step = azimuths[1] - azimuths[0]
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    return cart_img


# ============================================================
# 主函数：批量读取目录下所有雷达 FFT npy 文件并转存
# ============================================================
def convert_radar_dir_to_cartesian_from_fft_npy(input_dir: str,
                                                output_dir: str,
                                                radar_resolution: float = 0.0432,
                                                cart_resolution: float = 0.25,
                                                cart_pixel_width: int = 501,
                                                save_as_png: bool = False):
    """
    读取 input_dir 下的所有 FFT npy 文件 (二维数组)，
    转换为笛卡尔坐标并存储到 output_dir。
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"Found {len(files)} FFT npy files in {input_dir}")

    for i, path in enumerate(files):
        fft_data = np.load(path).astype(np.float32)

        cart_img = radar_polar_to_cartesian(
            fft_data,
            radar_resolution,
            cart_resolution,
            cart_pixel_width
        )

        cart_img = cart_img.squeeze()
        base = os.path.splitext(os.path.basename(path))[0]

        if save_as_png:
            out_path = os.path.join(output_dir, f"{base}_cart.png")
            norm = cv2.normalize(cart_img, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(out_path, norm.astype(np.uint8))
        else:
            out_path = os.path.join(output_dir, f"{base}_cart.npy")
            np.save(out_path, cart_img)

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"[{i+1}/{len(files)}] Saved: {out_path}")

    print(f"\n✅ Conversion finished. Output saved to: {output_dir}")


# ============================================================
# 使用示例
# ============================================================
folder = Path("temps/")
for item in folder.iterdir():
    if item.is_file():
        item.unlink()              # 删除文件
    elif item.is_dir():
        shutil.rmtree(item)        # 删除子目录及其内容 
if __name__ == "__main__":
    convert_radar_dir_to_cartesian_from_fft_npy(
        input_dir="preprocess_results/thresholded_fft/seq10/",          # 输入 FFT npy 文件目录
        # input_dir="preprocess_results/occupancy_component/preprocess_results/",          # 输入 FFT npy 文件目录
        output_dir="temps/",    # 输出笛卡尔结果目录
        radar_resolution=0.0432,                  # 雷达距离分辨率（米）
        cart_resolution=0.25,                     # 每像素代表 0.25 米
        cart_pixel_width=501,                     # 输出图像尺寸
        save_as_png=False                         # True: 保存 png; False: 保存 npy
    )
