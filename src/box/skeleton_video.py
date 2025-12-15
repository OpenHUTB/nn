import os
import sys
import numpy as np
import mujoco
import imageio
from PIL import Image
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告

# 配置项目路径（关键：指向user-in-the-box的模型目录）
PROJECT_ROOT = "D:/nn/user-in-the-box-main/user-in-the-box-main"
sys.path.append(PROJECT_ROOT)
MODEL_PATH = os.path.join(PROJECT_ROOT, "figs/mobl_arms_index/mobl_arms_wrist.xml")

# 校验模型文件是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}\n请检查 PROJECT_ROOT 路径是否正确")

# -------------------------- 视频/GIF配置 --------------------------
VIDEO_DURATION = 8  # 运动时长（秒）
FPS = 30  # 仿真帧率
GIF_FPS = 10  # GIF帧率
OUTPUT_VIDEO = "arm_tracking_video.mp4"
OUTPUT_GIF = "video3.gif"
frames = []  # 存储帧

# -------------------------- 初始化MuJoCo仿真 --------------------------
# 加载手臂模型
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 使用Mujoco官方内置Viewer（替代mujoco_viewer）
viewer = mujoco.viewer.launch_passive(model, data, width=800, height=600)

# 定义手臂运动轨迹（模拟追踪目标的屈伸+旋转）
def set_arm_motion(data, time_ratio):
    """按时间比例控制关节角度，模拟自然追踪运动"""
    try:
        # 肩关节旋转（绕Y轴）
        data.joint("shoulder_rot").qpos = np.sin(time_ratio * 2 * np.pi) * 0.5
        # 肘关节弯曲（0=伸直，1.5=90°左右）
        data.joint("elbow_flex").qpos = (1 - np.cos(time_ratio * 2 * np.pi)) * 1.2
        # 腕关节偏移（跟随肘关节联动）
        data.joint("wrist_dev").qpos = np.sin(time_ratio * 3 * np.pi) * 0.3
        # 目标追踪点（彩色小球，模拟video3.gif的标记点）
        target_body = data.body("target_ball")
        target_body.xpos = [
            0.5 + np.sin(time_ratio * 2 * np.pi) * 0.3,
            0.2 - np.cos(time_ratio * 2 * np.pi) * 0.2,
            0.1
        ]
    except Exception as e:
        print(f"\n关节/刚体名称错误：{e}")
        print("请检查模型文件中的关节/刚体命名是否匹配")

# -------------------------- 运行仿真+录制帧 --------------------------
total_frames = VIDEO_DURATION * FPS
# 初始化渲染上下文
mujoco.mj_forward(model, data)

for frame_idx in range(total_frames):
    # 计算时间进度（0~1）
    time_ratio = frame_idx / total_frames
    
    # 更新手臂运动
    set_arm_motion(data, time_ratio)
    
    # 步进仿真
    mujoco.mj_step(model, data)
    
    # 同步viewer并捕获帧（官方API方式）
    viewer.sync()
    # 读取像素（height, width, 3），自动处理RGB格式
    frame = mujoco.mjr_readPixels(model, data, width=800, height=600)
    # 转换为uint8格式（避免GIF颜色异常）
    frame = (frame * 255).astype(np.uint8)
    frames.append(frame)
    
    # 打印进度
    print(f"录制进度：{frame_idx+1}/{total_frames} 帧", end="\r")

# -------------------------- 释放资源 --------------------------
viewer.close()

# -------------------------- 生成MP4视频 --------------------------
try:
    imageio.mimsave(
        OUTPUT_VIDEO,
        frames,
        fps=FPS,
        codec="libx264",
        quality=9,
        pixelformat="yuv420p"
    )
except Exception as e:
    print(f"\nMP4生成失败：{e}")
    # 降级方案：生成无压缩AVI
    imageio.mimsave("arm_tracking_video.avi", frames, fps=FPS)
    print("已生成降级版AVI视频：arm_tracking_video.avi")

# -------------------------- 转换为GIF（匹配video3.gif格式） --------------------------
try:
    # 优化：每隔3帧取1帧，缩小尺寸，降低GIF体积
    gif_frames = []
    for i in range(0, len(frames), 3):
        img = Image.fromarray(frames[i])
        img = img.resize((400, 300), Image.Resampling.LANCZOS)  # 高质量缩放
        gif_frames.append(np.array(img))
    
    # 生成GIF（无限循环）
    imageio.mimsave(
        OUTPUT_GIF,
        gif_frames,
        fps=GIF_FPS,
        loop=0,
        palettesize=256  # 适配GIF颜色深度
    )
except Exception as e:
    print(f"\nGIF生成失败：{e}")
    print("请检查imageio版本：pip install imageio-ffmpeg --upgrade")

# -------------------------- 输出结果 --------------------------
print(f"\n===== 生成完成 =====")
if os.path.exists(OUTPUT_VIDEO):
    print(f"视频文件：{os.path.abspath(OUTPUT_VIDEO)}")
if os.path.exists(OUTPUT_GIF):
    print(f"GIF文件：{os.path.abspath(OUTPUT_GIF)}")
    