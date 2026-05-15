import cv2
import os
import glob

# 读取图片路径
img_paths = sorted(glob.glob("_out/*.png"))

# 视频设置（关键：全部调低 = 视频变小）
fps = 10          # 帧率调低
width = 340       # 宽度减半
height = 210      # 高度减半
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 输出视频
out = cv2.VideoWriter("lidar_short.mp4", fourcc, fps, (width, height))

# 只合成前 80 帧（控制时长）
for i, path in enumerate(img_paths[:80]):
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    out.write(img)
    print(f"合成第 {i+1} 帧")

out.release()
print("\n✅ 视频生成完成：lidar_short.mp4")
print("📦 大小大约 3~5MB")

