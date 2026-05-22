
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "当前版本（v1.0）已完美验证了静态感知闭环的可行性，具备了作为优秀开源基础模块的所有要素。"
new = "当前版本（v1.0）已充分验证了静态感知闭环的可行性，具备了作为优秀开源基础模块的各项要素。"
t = t.replace(old, new, 1)

old2 = "**现状**：目前依赖 YOLOv8 在 COCO 数据集上的预训练权重，仅能识别 `Stop Sign` 类别，且对 **中国国标交通标志**（如注意行人、限速标志）不具备识别能力。"
new2 = "**现状**：目前依赖 YOLOv8 在 COCO 数据集上的预训练权重，仅能识别 `Stop Sign` 单一类别，且对 **中国国标交通标志**（如注意行人、限速标志等）不具备识别能力。"
t = t.replace(old2, new2, 1)

old3 = "**规划**：单目视觉缺乏绝对的深度信息且易受恶劣天气干扰。后续将在 Carla 中为车辆挂载 **LiDAR（激光雷达）**。"
new3 = "**规划**：单目视觉缺乏绝对的深度信息，且易受恶劣天气干扰。后续将在 Carla 中为车辆挂载 **LiDAR（激光雷达）** 传感器。"
t = t.replace(old3, new3, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
