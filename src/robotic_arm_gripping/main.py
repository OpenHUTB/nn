import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import sys

# 全局控制变量
key_pressed = {
    'w': False, 's': False, 'a': False, 'd': False,
    'q': False, 'e': False, 'space': False,
    'auto': True  # 默认自动模式
}
action_speed = 0.08  # 动作速度（越大越快）
grab_threshold = 0.12  # 抓取判定阈值
ball_drop_threshold = 0.05  # 小球脱落判定阈值

def key_listener():
    """键盘监听线程（非阻塞）"""
    import keyboard
    def on_press(key):
        try:
            if key.name in key_pressed:
                key_pressed[key.name] = True
            elif key.name == 'esc':
                sys.exit(0)
        except:
            pass

    def on_release(key):
        try:
            if key.name in key_pressed:
                key_pressed[key.name] = False
        except:
            pass

    keyboard.on_press(on_press)
    keyboard.on_release(on_release)
    while True:
        time.sleep(0.01)

def get_ball_position(data, model):
    """获取目标小球位置"""
    ball_body_id = model.body('target').id
    return data.xpos[ball_body_id].copy()

def is_ball_grabbed(data, model):
    """检测是否成功抓取小球"""
    ball_pos = get_ball_position(data, model)
    gripper_pos = data.xpos[model.body('wrist').id]
    distance = np.linalg.norm(ball_pos - gripper_pos)
    return distance < grab_threshold

def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path("arm_model.xml")
    data = mujoco.MjData(model)

    # 启动键盘监听线程
    threading.Thread(target=key_listener, daemon=True).start()

    # 初始化变量
    phase = 0
    phase_time = 0.0
    target = np.zeros(4)  # [肩, 肘, 左指, 右指]
    last_ball_pos = get_ball_position(data, model)
    auto_phase_durations = [2.0, 2.0, 1.0, 1.0, 3.0, 2.0, 2.0]  # 各阶段时长

    # 启动可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 相机参数优化
        viewer.cam.azimuth = 60
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat = [0.2, 0, 0.3]

        print("="*50)
        print("MuJoCo 两指机械臂控制程序")
        print("控制说明：")
        print("  W/S：控制肩部关节（上下）")
        print("  A/D：控制肘部关节（前后）")
        print("  Q/E：控制手指开合（抓取/释放）")
        print("  空格：重置初始姿态")
        print("  ESC：退出程序")
        print("  自动模式：默认开启，按任意键切换为手动模式")
        print("="*50)

        while viewer.is_running():
            current_time = time.time()
            phase_time += model.opt.timestep

            # 手动控制逻辑（优先级高于自动）
            manual_control = np.zeros(4)
            if any([key_pressed[k] for k in ['w','s','a','d','q','e','space']]):
                key_pressed['auto'] = False
                # 肩部控制（W=上，S=下）
                if key_pressed['w']:
                    manual_control[0] -= action_speed
                if key_pressed['s']:
                    manual_control[0] += action_speed
                # 肘部控制（A=前，D=后）
                if key_pressed['a']:
                    manual_control[1] -= action_speed
                if key_pressed['d']:
                    manual_control[1] += action_speed
                # 手指控制（Q=闭合，E=张开）
                if key_pressed['q']:
                    manual_control[2] -= action_speed
                    manual_control[3] -= action_speed
                if key_pressed['e']:
                    manual_control[2] += action_speed
                    manual_control[3] += action_speed
                # 重置姿态
                if key_pressed['space']:
                    target = np.array([0.2, -0.6, 0.0, 0.0])
                    phase = 0
                    phase_time = 0
                    key_pressed['auto'] = True  # 重置后恢复自动模式

                # 应用手动控制
                target += manual_control
                # 限制关节范围
                target[0] = np.clip(target[0], -1.57, 1.57)  # 肩部
                target[1] = np.clip(target[1], -2.0, 2.0)     # 肘部
                target[2] = np.clip(target[2], 0.0, 0.8)      # 左指
                target[3] = np.clip(target[3], 0.0, 0.8)      # 右指

            # 自动循环抓取逻辑
            elif key_pressed['auto']:
                # 阶段0：初始准备
                if phase == 0:
                    target = np.array([0.2, -0.6, 0.0, 0.0])
                    if phase_time > auto_phase_durations[phase]:
                        phase = 1
                        phase_time = 0
                        print("\r[自动模式] 阶段1：向前伸臂", end="")

                # 阶段1：向前伸臂
                elif phase == 1:
                    target = np.array([0.0, -1.2, 0.0, 0.0])
                    if phase_time > auto_phase_durations[phase]:
                        phase = 2
                        phase_time = 0
                        print("\r[自动模式] 阶段2：张开手指", end="")

                # 阶段2：张开手指
                elif phase == 2:
                    target = np.array([0.0, -1.2, 0.7, 0.7])
                    if phase_time > auto_phase_durations[phase]:
                        phase = 3
                        phase_time = 0
                        print("\r[自动模式] 阶段3：闭合抓取", end="")

                # 阶段3：闭合抓取
                elif phase == 3:
                    target = np.array([0.0, -1.2, 0.1, 0.1])
                    # 检测抓取状态，未抓到则延长该阶段
                    if is_ball_grabbed(data, model) or phase_time > auto_phase_durations[phase]*2:
                        phase = 4
                        phase_time = 0
                        print("\r[自动模式] 阶段4：举升小球", end="")

                # 阶段4：举升小球
                elif phase == 4:
                    target = np.array([-0.7, 0.6, 0.1, 0.1])
                    # 检测小球是否脱落，脱落则重新抓取
                    current_ball_pos = get_ball_position(data, model)
                    if np.linalg.norm(current_ball_pos - last_ball_pos) > ball_drop_threshold:
                        print("\r[警告] 小球脱落，重新抓取！", end="")
                        phase = 1
                        phase_time = 0
                    elif phase_time > auto_phase_durations[phase]:
                        phase = 5
                        phase_time = 0
                        print("\r[自动模式] 阶段5：放下小球", end="")
                    last_ball_pos = current_ball_pos

                # 阶段5：放下小球
                elif phase == 5:
                    target = np.array([0.0, -1.2, 0.1, 0.1])
                    if phase_time > auto_phase_durations[phase]:
                        phase = 6
                        phase_time = 0
                        print("\r[自动模式] 阶段6：张开手指", end="")

                # 阶段6：张开手指复位
                elif phase == 6:
                    target = np.array([0.0, -1.2, 0.7, 0.7])
                    if phase_time > auto_phase_durations[phase]:
                        phase = 0
                        phase_time = 0
                        print("\r[自动模式] 阶段0：复位初始姿态", end="")

            # 平滑插值控制（避免动作突变）
            data.ctrl[:] = 0.95 * data.ctrl + 0.05 * target

            # 执行物理仿真步
            mujoco.mj_step(model, data)
            viewer.sync()

            # 控制帧率（约120fps）
            time.sleep(max(0, 1/120 - (time.time() - current_time)))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序正常退出")
    except Exception as e:
        print(f"\n程序异常：{e}")