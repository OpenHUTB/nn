"""
[路径优化] 自动切换工作目录并加载模型

为了增强代码的可移植性，我们在程序启动时自动将工作目录切换到项目根目录。
这样可以确保脚本中使用的相对路径（如 "src/humanoid_motion_control/humanoid.xml"）
无论从哪个位置运行，都能正确地指向预期的文件。

修改原因：
- 原版代码使用的是基于运行时目录的相对路径，这在不同环境或不同运行方式下容易出错。
- 通过自动切换到项目根目录，可以统一所有相对路径的基准，避免 "No such file or directory" 错误。

使用方法：
- 保持 main.py 中原有相对路径写法不变即可。
- 此脚本会在加载模型等操作前自动完成路径切换。

注意事项：
- 此代码段必须放置在所有依赖相对路径的导入或文件操作之前。
- 如果项目目录结构发生重大改变，可能需要调整计算 project_root 的层级。
"""

import os
import mujoco

# --- 你之前的切换目录代码 ---
import sys
file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(file_path))
os.chdir(project_root)
print(f"当前工作目录已切换到: {os.getcwd()}")
# ----------------------------

# 1. 定义模型文件的相对路径
# 注意：这里使用了推荐的正斜杠
relative_model_path = "src/humanoid_motion_control/humanoid.xml"
print(f"尝试查找的模型路径: {relative_model_path}")

# 2. 检查文件是否存在
if os.path.exists(relative_model_path):
    print("✅ 模型文件找到了！")
    try:
        # 3. 如果文件存在，再尝试加载
        model = mujoco.MjModel.from_xml_path(relative_model_path)
        data = mujoco.MjData(model)
        print("🎉 模型加载成功！")
        # ... 你的其他代码 ...
    except Exception as e:
        print(f"模型加载失败: {e}")
else:
    print(f"❌ 错误：在当前工作目录下找不到模型文件。")
    print(f"请确认文件 '{relative_model_path}' 确实存在。")


# 标准库
import time
# 第三方库
import mujoco
from mujoco import viewer

def main():
    try:
        model = mujoco.MjModel.from_xml_path(r"humanoid_motion_control\humanoid.xml")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    data = mujoco.MjData(model)
    # 临时数据用于获取目标关键帧（站立姿势，索引1）
    target_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, target_data, 1) 
    target_qpos = target_data.qpos.copy()  
    
    # 初始姿势设为深蹲（索引0），从蹲下开始站起
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 控制参数（软体机器人需较小增益，避免震荡）
    Kp = 5.0  # 比例增益（根据软体特性调小）
    
    with viewer.launch_passive(model, data) as v:
        try:
            while True:
                mujoco.mj_step(model, data)
                
                # 计算关节位置误差（跳过前7个根关节，只控制电机关节）
                qpos_error = target_qpos[7:] - data.qpos[7:]
                # 比例控制：控制信号与误差成正比（适配软体机器人的柔性特性）
                data.ctrl[:] = Kp * qpos_error
                
                print(f"时间: {data.time:.2f}, 躯干高度: {data.qpos[2]:.2f}")
                
                v.sync()
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\n正在退出模拟....")

if __name__ == "__main__":
    main()