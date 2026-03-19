import cv2
import glob
import time

# 读取所有 PNG 图片，按文件名排序
image_files = sorted(glob.glob("workspace/imgs/radarfields/FFT/*.png"))
# image_files = sorted(glob.glob("workspace/imgs/radarfields/pred_FFT/*.png"))
# image_files = sorted(glob.glob("workspace/imgs/radarfields/40/pred_occupancy/*.png"))


# 设置播放间隔（秒）
fps = 10  # 每秒播放10帧
delay = 1 / fps

for img_path in image_files:
    img = cv2.imread(img_path)
    img = img[:, 11:1500]  # 截取有效区域
    cv2.imshow("Playback", img)

    # 等待 delay 秒或直到按下 ESC 退出
    if cv2.waitKey(int(delay * 1000)) & 0xFF == 27:
        break

cv2.destroyAllWindows()
