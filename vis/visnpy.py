import numpy as np
import cv2
import glob
import time

# 读取所有 npy 文件
npy_files = sorted(glob.glob("data/preprocess_results/*.npy"))
# npy_files = sorted(glob.glob("data/preprocess_results_raw/*.npy"))
npy_files = sorted(glob.glob("preprocess_results/occupancy_component/preprocess_results/*.npy"))
npy_files = sorted(glob.glob("preprocess_results/thresholded_fft/seq10/*.npy"))
npy_files = sorted(glob.glob("temps/*.npy"))

fps = 10  # 每秒播放帧数
delay = 1 / fps

for f in npy_files:
    frame = np.load(f)  # 加载一帧 (H, W) 或 (H, W, 3)

    # 如果是单通道图像，转为8位灰度显示
    if frame.ndim == 2:
        img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 可视化用伪彩色
    else:
        img = (frame * 255).astype(np.uint8)

    cv2.imshow("NPY Playback", img)
    if cv2.waitKey(int(delay * 1000)) & 0xFF == 27:
        break

cv2.destroyAllWindows()
