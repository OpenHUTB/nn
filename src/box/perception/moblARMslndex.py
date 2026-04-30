"""
MoblArmsIndex 简化版本 - 只生成人体上肢模型运动视频
完全修复版
"""

import numpy as np
import mujoco
import imageio
import os
import subprocess
import sys

def install_ffmpeg():
    """安装必要的视频编码依赖"""
    try:
        # 尝试导入 imageio-ffmpeg
        import imageio_ffmpeg
        print("✓ imageio-ffmpeg 已安装")
        return True
    except ImportError:
        print("正在安装 imageio-ffmpeg...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio[ffmpeg]"])
            print("✓ imageio[ffmpeg] 安装成功")
            return True
        except Exception as e:
            print(f"⚠ 安装失败: {e}")
            print("将使用 GIF 格式替代 MP4")
            return False

# ================ 简化的 BaseBMModel 类 ================
class BaseBMModel:
    """简化的 BaseBMModel 基类"""
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data
        
    def _update(self, model, data):
        """更新方法 - 子类应重写"""
        pass

# ================ MoblArmsIndex 模型 ================
class MoblArmsIndex(BaseBMModel):
    """
    基于 MoBL ARMS 模型的生物力学模型
    只专注于生成上肢运动视频
    """
    
    def __init__(self, model, data, **kwargs):
        super().__init__(model, data, **kwargs)
        print("✓ MoblArmsIndex 上肢模型初始化完成")

    def _update(self, model, data):
        """更新模型状态 - 简化版本"""
        # 这里可以添加任何必要的模型更新逻辑
        pass

# ================ 主程序：只生成视频 ================
def main():
    print("=" * 60)
    print("人体上肢模型运动视频生成器")
    print("=" * 60)
    
    # 检查并安装必要依赖
    has_ffmpeg = install_ffmpeg()
    
    # 1. 创建更真实的上肢XML模型（修复了所有问题）
    xml_string = """
    <mujoco>
        <option timestep="0.01" iterations="50"/>
        
        <visual>
            <global azimuth="45" elevation="-20" offwidth="640" offheight="480"/>
        </visual>
        
        <worldbody>
            <!-- 简单灯光 -->
            <light name="light1" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            
            <!-- 地面 -->
            <geom name="floor" type="plane" pos="0 0 0" size="2 2 0.1" rgba="0.95 0.95 0.95 1"/>
            
            <!-- 人体躯干 -->
            <body name="torso" pos="0 0 1.0">
                <geom type="capsule" fromto="0 0 0.1 0 0 0.4" size="0.12" rgba="0.4 0.4 0.6 1"/>
                
                <!-- 右肩部 -->
                <body name="right_shoulder" pos="0.15 0 0.25">
                    <geom type="sphere" size="0.08" rgba="0.5 0.5 0.7 1"/>
                    
                    <!-- 上臂 -->
                    <body name="right_upper_arm" pos="0 0 0">
                        <joint name="shoulder_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                        <joint name="shoulder_roll" type="hinge" axis="1 0 0" range="-1.0 1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.06" rgba="0.6 0.6 0.8 1"/>
                        
                        <!-- 肘部 -->
                        <body name="right_forearm" pos="0 0 0.35">
                            <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.05" rgba="0.7 0.7 0.9 1"/>
                            
                            <!-- 手腕 -->
                            <body name="right_hand" pos="0 0 0.3">
                                <joint name="wrist" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                                <geom type="box" size="0.04 0.06 0.02" rgba="0.8 0.8 1.0 1"/>
                                
                                <!-- 食指 -->
                                <body name="right_index" pos="0.04 0 0.02">
                                    <joint name="index_joint" type="hinge" axis="0 1 0" range="-1.0 0.2"/>
                                    <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.012" rgba="1.0 0.3 0.3 1"/>
                                </body>
                                
                                <!-- 拇指 -->
                                <body name="right_thumb" pos="0 -0.04 0.02">
                                    <joint name="thumb_joint" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                                    <geom type="capsule" fromto="0 0 0 0.04 0 0.04" size="0.01" rgba="1.0 0.6 0.3 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
                
                <!-- 左臂（保持静止） -->
                <body name="left_shoulder" pos="-0.15 0 0.25">
                    <geom type="sphere" size="0.08" rgba="0.5 0.5 0.7 1"/>
                    
                    <body name="left_upper_arm" pos="0 0 0">
                        <geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.06" rgba="0.6 0.6 0.8 1"/>
                        
                        <body name="left_forearm" pos="0 0 0.35">
                            <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.05" rgba="0.7 0.7 0.9 1"/>
                            
                            <body name="left_hand" pos="0 0 0.3">
                                <geom type="box" size="0.04 0.06 0.02" rgba="0.8 0.8 1.0 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- 目标点 -->
            <body name="target" pos="0.5 0.3 1.5">
                <geom type="sphere" size="0.03" rgba="0 1 0 0.7"/>
            </body>
        </worldbody>
        
        <actuator>
            <!-- 右臂控制 -->
            <motor name="shoulder_pitch_motor" joint="shoulder_pitch" gear="80"/>
            <motor name="shoulder_roll_motor" joint="shoulder_roll" gear="60"/>
            <motor name="elbow_motor" joint="elbow" gear="100"/>
            <motor name="wrist_motor" joint="wrist" gear="40"/>
            <motor name="index_motor" joint="index_joint" gear="30"/>
            <motor name="thumb_motor" joint="thumb_joint" gear="20"/>
        </actuator>
    </mujoco>
    """
    
    # 2. 加载模型
    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        print("✓ MuJoCo 模型加载成功")
        print(f"  关节数: {model.njnt}")
        print(f"  几何体数: {model.ngeom}")
        print(f"  执行器数: {model.nu}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 3. 创建生物力学模型实例
    bm_model = MoblArmsIndex(model, data)
    
    # 4. 设置仿真参数
    fps = 30
    duration = 5  # 5秒仿真
    total_steps = int(duration / model.opt.timestep)
    
    print(f"\n仿真参数:")
    print(f"  时长: {duration}秒")
    print(f"  时间步长: {model.opt.timestep}")
    print(f"  总步数: {total_steps}")
    print(f"  视频帧率: {fps} FPS")
    
    # 5. 设置渲染器（使用较小的分辨率避免问题）
    try:
        # 创建渲染器（使用640x480避免离屏缓冲区问题）
        renderer = mujoco.Renderer(model, height=480, width=640)
        
        # 设置相机视角
        camera = mujoco.MjvCamera()
        camera.azimuth = 30      # 水平角度
        camera.elevation = -20   # 垂直角度
        camera.distance = 2.5    # 距离
        camera.lookat = np.array([0.2, 0, 1.2])  # 看向点
        
        # 初始渲染
        renderer.update_scene(data, camera=camera)
        print("✓ 渲染器创建成功")
    except Exception as e:
        print(f"❌ 渲染器创建失败: {e}")
        return
    
    # 6. 运行仿真并捕获视频帧
    frames = []
    
    print("\n开始仿真并录制视频...")
    
    for step in range(total_steps):
        # 计算时间（用于控制运动）
        t = step * model.opt.timestep
        
        # 协同运动控制
        # 肩部运动（缓慢的波浪）
        shoulder_pitch = 0.3 * np.sin(2 * np.pi * 0.3 * t) + 0.1
        shoulder_roll = 0.2 * np.sin(2 * np.pi * 0.2 * t + 0.5)
        
        # 肘部运动
        elbow = 0.6 * np.sin(2 * np.pi * 0.4 * t) - 1.0
        
        # 手腕运动
        wrist = 0.1 * np.sin(2 * np.pi * 0.6 * t)
        
        # 手指运动序列
        if t < 1:
            # 初始：手指弯曲
            index_angle = -0.4
            thumb_angle = 0.1
        elif t < 3:
            # 指向阶段
            phase = (t - 1) / 2
            index_angle = 0.1 * np.sin(2 * np.pi * 1.0 * t) - 0.1
            thumb_angle = 0.15 * np.sin(2 * np.pi * 0.8 * t)
        else:
            # 返回阶段
            phase = (t - 3) / 2
            index_angle = -0.4 * (1 - phase)
            thumb_angle = 0.1 * (1 - phase)
        
        # 应用控制信号
        data.ctrl[0] = shoulder_pitch    # 肩部俯仰
        data.ctrl[1] = shoulder_roll     # 肩部滚动
        data.ctrl[2] = elbow             # 肘部
        data.ctrl[3] = wrist             # 手腕
        data.ctrl[4] = index_angle       # 食指
        data.ctrl[5] = thumb_angle       # 拇指
        
        # 步进仿真
        mujoco.mj_step(model, data)
        
        # 定期捕获帧
        if step % 2 == 0:  # 每2步捕获一帧，获得15fps（GIF比较流畅）
            try:
                # 动态调整相机视角
                if t < 2:
                    camera.azimuth = 30 + 15 * t
                elif t < 4:
                    camera.elevation = -20 + 10 * np.sin(t * 0.5)
                
                # 更新场景并渲染
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                frames.append(frame)
            except:
                # 如果失败，使用默认视角
                try:
                    renderer.update_scene(data)
                    frame = renderer.render()
                    frames.append(frame)
                except:
                    pass
        
        # 显示进度
        if step % 100 == 0:
            progress = (step / total_steps) * 100
            print(f"进度: {progress:.1f}%", end='\r')
    
    print(f"进度: 100.0% - 仿真完成!")
    
    # 7. 生成视频文件
    if frames:
        try:
            # 确定保存路径（桌面）
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            if not os.path.exists(desktop):
                desktop = os.path.dirname(os.path.abspath(__file__))
            
            # 首先尝试生成MP4（如果有FFMPEG）
            if has_ffmpeg:
                try:
                    video_path = os.path.join(desktop, "human_arm_motion.mp4")
                    
                    # 保存MP4视频
                    print(f"\n生成MP4视频中...")
                    imageio.mimsave(
                        video_path,
                        frames,
                        fps=15,  # GIF帧率较低
                        codec='libx264',
                        quality=8
                    )
                    print(f"✓ MP4视频已保存: {video_path}")
                    print(f"  文件大小: {os.path.getsize(video_path)/1024/1024:.1f} MB")
                    
                except Exception as mp4_error:
                    print(f"⚠ MP4生成失败: {mp4_error}")
                    # 回退到GIF
                    has_ffmpeg = False
            
            # 如果没有FFMPEG或MP4生成失败，生成GIF
            if not has_ffmpeg:
                gif_path = os.path.join(desktop, "human_arm_motion.gif")
                
                print(f"\n生成GIF动画中...")
                # 保存为GIF（质量稍低但兼容性好）
                imageio.mimsave(
                    gif_path,
                    frames,
                    fps=15,  # GIF帧率较低
                    subrectangles=True  # 优化GIF大小
                )
                print(f"✓ GIF动画已保存: {gif_path}")
                print(f"  文件大小: {os.path.getsize(gif_path)/1024:.1f} KB")
                
        except Exception as e:
            print(f"❌ 文件保存失败: {e}")
    else:
        print("⚠ 没有捕获到视频帧")
    
    print("\n" + "=" * 60)
    print("✅ 任务完成!")
    print("=" * 60)
    
    # 8. 打开文件所在目录（仅Windows）
    if 'video_path' in locals() and os.path.exists(video_path):
        print(f"📁 MP4视频文件: {video_path}")
        try:
            os.startfile(os.path.dirname(video_path))
            print("📂 已打开文件所在目录")
        except:
            pass
    elif 'gif_path' in locals() and os.path.exists(gif_path):
        print(f"📁 GIF动画文件: {gif_path}")
        try:
            os.startfile(os.path.dirname(gif_path))
            print("📂 已打开文件所在目录")
        except:
            pass

# ================ 运行主程序 ================
if __name__ == "__main__":
    # 检查必要依赖
    try:
        import mujoco
        import numpy as np
        import imageio
        print("✓ 核心库已安装")
    except ImportError as e:
        print(f"❌ 缺少必要的库: {e}")
        print("\n请安装以下依赖:")
        print("pip install mujoco numpy imageio")
        exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ 用户中断")
    except Exception as e:
        print(f"\n❌ 运行出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        