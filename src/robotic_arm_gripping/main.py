# 导入必要的库
import mujoco  # MuJoCo仿真核心库，用于加载模型和运行物理仿真
import mujoco.viewer  # MuJoCo可视化模块，用于显示仿真界面
import numpy as np  # 数值计算库，用于矩阵/向量运算
import time  # 时间控制库，用于帧率控制和计时
import threading  # 多线程库，用于启动非阻塞的键盘监听
import sys  # 系统库，用于退出程序和处理异常
import keyboard  # 键盘监听库，用于捕获按键事件

# ===================== 全局配置区（所有可调整参数集中管理） =====================
# 全局按键状态字典：记录每个按键的按下/释放状态，初始均为False（未按下）
KEY_STATE = {
    'w': False, 's': False, 'a': False, 'd': False,  # 方向/关节控制键
    'q': False, 'e': False,  # 夹爪控制键
    'space': False,  # 重置姿态键
    'auto': True  # 自动/手动模式切换标记（默认自动）
}

# 核心控制参数配置（所有参数可在此调整，无需修改核心逻辑）
CFG = {
    'speed': 0.08,  # 手动控制时的动作速度（值越大，关节运动越快）
    'grab_thresh': 0.12,  # 抓取判定阈值（夹爪与小球距离小于此值则判定为抓取成功）
    'drop_thresh': 0.05,  # 小球脱落阈值（举升时小球位移超过此值则判定为脱落）
    # 关节运动限位（防止关节超出物理范围）
    'shoulder_limit': (-1.57, 1.57),  # 肩部关节角度范围（对应-90°到90°）
    'elbow_limit': (-2.0, 2.0),  # 肘部关节角度范围
    'finger_limit': (0.0, 0.8),  # 夹爪关节角度范围（0=闭合，0.8=完全张开）
    # 机械臂初始姿态（[肩部, 肘部, 左夹爪, 右夹爪]）
    'init_pose': np.array([0.2, -0.6, 0.0, 0.0]),
    # 相机参数（控制仿真界面的视角）
    'cam': {
        'azimuth': 60,  # 相机方位角（水平旋转角度）
        'elevation': -25,  # 相机仰角（垂直旋转角度，负值为俯视）
        'distance': 1.8,  # 相机与机械臂的距离
        'lookat': [0.2, 0, 0.3]  # 相机聚焦点（机械臂基座附近）
    }
}


# ===================== 工具函数区（封装通用功能，便于复用） =====================
def key_listener():
    """
    键盘监听线程函数（非阻塞）
    作用：持续监听按键的按下/释放事件，更新KEY_STATE字典的状态
    特性：后台运行，不阻塞主仿真循环
    """

    # 按键按下时的回调函数
    def on_press(key):
        try:
            # 将按键名转为小写，避免大小写敏感问题
            k = key.name.lower()
            # 如果是我们关注的按键，标记为按下状态
            if k in KEY_STATE:
                KEY_STATE[k] = True
            # 按ESC键直接退出程序
            elif k == 'esc':
                sys.exit(0)
        except Exception:
            # 忽略按键识别异常（如特殊键）
            pass

    # 按键释放时的回调函数
    def on_release(key):
        try:
            k = key.name.lower()
            # 如果是我们关注的按键，标记为释放状态
            if k in KEY_STATE:
                KEY_STATE[k] = False
        except Exception:
            pass

    # 先解绑所有已注册的键盘监听（防止重复监听）
    keyboard.unhook_all()
    # 注册按键按下/释放的回调函数
    keyboard.on_press(on_press)
    keyboard.on_release(on_release)
    # 线程循环（持续运行，睡眠减少CPU占用）
    while True:
        time.sleep(0.001)


def get_ball_pos(data, model):
    """
    获取目标小球的三维坐标
    参数：
        data: MuJoCo的数据对象（存储仿真过程中的实时数据）
        model: MuJoCo的模型对象（存储模型的结构信息）
    返回：
        np.ndarray: 小球的[x, y, z]坐标，获取失败则返回[0,0,0]
    """
    try:
        # 通过模型获取小球body的ID，再从data中读取其位置
        ball_body_id = model.body('target').id
        return data.xpos[ball_body_id].copy()
    except Exception:
        # 容错处理：获取失败时返回全0坐标
        return np.zeros(3)


def is_grabbed(data, model):
    """
    判断是否成功抓取小球
    原理：计算夹爪腕部（wrist）与小球的直线距离，小于阈值则判定为抓取成功
    参数：
        data: MuJoCo数据对象
        model: MuJoCo模型对象
    返回：
        bool: True=抓取成功，False=未抓取
    """
    # 获取小球和夹爪腕部的坐标
    ball_pos = get_ball_pos(data, model)
    gripper_pos = data.xpos[model.body('wrist').id]
    # 计算欧氏距离（直线距离）
    distance = np.linalg.norm(ball_pos - gripper_pos)
    # 距离小于阈值则判定为抓取成功
    return distance < CFG['grab_thresh']


def clamp_joint(target):
    """
    关节角度限位函数
    作用：确保每个关节的目标角度不超出预设的物理范围，防止仿真出错
    参数：
        target: 待限位的关节角度数组 [肩部, 肘部, 左夹爪, 右夹爪]
    返回：
        np.ndarray: 限位后的关节角度数组
    """
    # 肩部关节限位
    target[0] = np.clip(target[0], *CFG['shoulder_limit'])
    # 肘部关节限位
    target[1] = np.clip(target[1], *CFG['elbow_limit'])
    # 左夹爪限位
    target[2] = np.clip(target[2], *CFG['finger_limit'])
    # 右夹爪限位
    target[3] = np.clip(target[3], *CFG['finger_limit'])
    return target


# ===================== 主函数区（核心仿真逻辑） =====================
def main():
    """
    程序主函数
    流程：加载模型 → 启动键盘监听 → 初始化变量 → 启动可视化 → 主仿真循环
    """
    # 1. 加载MuJoCo模型文件
    try:
        # 从指定路径加载模型（需确保arm_model.xml文件存在）
        model = mujoco.MjModel.from_xml_path("arm_model.xml")
        # 创建仿真数据对象（存储仿真过程中的实时状态）
        data = mujoco.MjData(model)
    except Exception:
        # 模型加载失败时提示并退出程序
        print("❌ 模型加载失败！请检查arm_model.xml文件路径是否正确。")
        sys.exit(1)

    # 2. 启动键盘监听线程（守护线程，主程序退出时自动结束）
    # target: 线程执行的函数；daemon=True: 设置为守护线程
    threading.Thread(target=key_listener, daemon=True).start()

    # 3. 初始化仿真变量
    phase = 0  # 自动模式的阶段标记（0-6）
    phase_time = 0  # 当前阶段的累计运行时间
    target = CFG['init_pose'].copy()  # 关节目标角度（初始化为初始姿态）
    last_ball_pos = get_ball_pos(data, model)  # 小球初始位置（用于检测脱落）

    # 4. 定义自动模式的动作序列（每个阶段对应一组关节目标角度）
    auto_actions = [
        np.array([0.2, -0.6, 0.0, 0.0]),  # 阶段0：复位到初始姿态
        np.array([0.0, -1.2, 0.0, 0.0]),  # 阶段1：向前伸展机械臂（接近小球）
        np.array([0.0, -1.2, 0.7, 0.7]),  # 阶段2：张开夹爪（准备抓取）
        np.array([0.0, -1.2, 0.1, 0.1]),  # 阶段3：闭合夹爪（抓取小球）
        np.array([-0.7, 0.6, 0.1, 0.1]),  # 阶段4：举升小球（向上移动）
        np.array([0.0, -1.2, 0.1, 0.1]),  # 阶段5：放下小球（回到初始高度）
        np.array([0.0, -1.2, 0.7, 0.7])  # 阶段6：张开夹爪（释放小球，复位）
    ]
    # 自动模式各阶段的持续时长（单位：秒）
    phase_durs = [2.0, 2.0, 1.0, 1.0, 3.0, 2.0, 2.0]
    # 自动模式各阶段的名称（用于控制台提示）
    phase_names = ["复位", "伸臂", "张手", "抓取", "举升", "放下", "复位"]

    # 5. 启动MuJoCo可视化界面（被动模式，需手动调用sync更新）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机参数（控制可视化视角）
        cam = viewer.cam
        cam.azimuth = CFG['cam']['azimuth']  # 水平角度
        cam.elevation = CFG['cam']['elevation']  # 垂直角度
        cam.distance = CFG['cam']['distance']  # 相机距离
        cam.lookat = CFG['cam']['lookat']  # 相机聚焦点

        # 打印控制说明（控制台提示用户操作方式）
        print("=" * 50)
        print("📌 机械臂控制程序 - 操作说明")
        print("  W/S：控制肩部关节（上下） | A/D：控制肘部关节（前后）")
        print("  Q/E：控制夹爪开合（Q=闭合/抓取，E=张开/释放）")
        print("  空格：重置为初始姿态（并恢复自动模式）")
        print("  ESC：退出程序")
        print("=" * 50)

        # 6. 主仿真循环（持续运行直到关闭可视化窗口）
        while viewer.is_running():
            # 累计当前阶段的运行时间（每次仿真步长累加）
            phase_time += model.opt.timestep
            # 初始化手动控制增量（默认无手动控制）
            manual_ctrl = np.zeros(4)

            # 7. 手动控制逻辑（有按键按下时触发）
            # 检测是否有手动控制按键被按下
            manual_trigger = any(KEY_STATE[k] for k in ['w', 's', 'a', 'd', 'q', 'e', 'space'])
            if manual_trigger:
                # 切换为手动模式
                KEY_STATE['auto'] = False

                # 肩部关节控制（W=向上，S=向下）
                if KEY_STATE['w']:
                    manual_ctrl[0] -= CFG['speed']  # W键：肩部角度减小（向上）
                if KEY_STATE['s']:
                    manual_ctrl[0] += CFG['speed']  # S键：肩部角度增大（向下）

                # 肘部关节控制（A=向前，D=向后）
                if KEY_STATE['a']:
                    manual_ctrl[1] -= CFG['speed']  # A键：肘部角度减小（向前）
                if KEY_STATE['d']:
                    manual_ctrl[1] += CFG['speed']  # D键：肘部角度增大（向后）

                # 夹爪控制（Q=闭合，E=张开）
                finger_ctrl = 0
                if KEY_STATE['q']:
                    finger_ctrl = -CFG['speed']  # Q键：夹爪角度减小（闭合）
                elif KEY_STATE['e']:
                    finger_ctrl = CFG['speed']  # E键：夹爪角度增大（张开）
                # 左右夹爪同步控制（保证对称）
                manual_ctrl[2] += finger_ctrl
                manual_ctrl[3] += finger_ctrl

                # 重置姿态（空格键）
                if KEY_STATE['space']:
                    target = CFG['init_pose'].copy()  # 恢复初始姿态
                    phase, phase_time = 0, 0  # 重置自动模式阶段
                    KEY_STATE['auto'] = True  # 恢复自动模式

                # 应用手动控制增量，并对关节角度限位
                target += manual_ctrl
                target = clamp_joint(target)

            # 8. 自动模式逻辑（无手动按键时运行）
            elif KEY_STATE['auto']:
                # 设置当前阶段的关节目标角度
                target = auto_actions[phase]

                # 阶段3（抓取）：特殊判断（抓取成功或超时则切换阶段）
                if phase == 3:
                    if is_grabbed(data, model) or phase_time > phase_durs[phase] * 2:
                        phase = (phase + 1) % 7  # 切换到下一个阶段（循环）
                        phase_time = 0  # 重置阶段计时
                        # 控制台提示当前阶段
                        print(f"\r[自动模式] {phase_names[phase]}", end="")

                # 阶段4（举升）：检测小球是否脱落
                elif phase == 4:
                    # 获取当前小球位置
                    curr_ball = get_ball_pos(data, model)
                    # 计算小球位移，超过阈值则判定为脱落
                    if np.linalg.norm(curr_ball - last_ball_pos) > CFG['drop_thresh']:
                        print("\r[警告] 小球脱落，重新抓取！", end="")
                        phase, phase_time = 1, 0  # 回到伸臂阶段重新抓取
                    # 未脱落且达到阶段时长则切换阶段
                    elif phase_time > phase_durs[phase]:
                        phase = (phase + 1) % 7
                        phase_time = 0
                        print(f"\r[自动模式] {phase_names[phase]}", end="")
                    # 更新小球位置（用于下次对比）
                    last_ball_pos = curr_ball

                # 其他阶段：按时长切换
                else:
                    if phase_time > phase_durs[phase]:
                        phase = (phase + 1) % 7
                        phase_time = 0
                        print(f"\r[自动模式] {phase_names[phase]}", end="")

            # 9. 平滑控制关节（避免角度突变导致仿真抖动）
            # 采用加权平均：95%当前角度 + 5%目标角度，实现平滑过渡
            data.ctrl[:] = 0.95 * data.ctrl + 0.05 * target

            # 10. 执行一次物理仿真步
            mujoco.mj_step(model, data)

            # 11. 更新可视化界面（同步仿真数据到界面）
            viewer.sync()

            # 12. 简单帧率控制（避免循环过快占用过多CPU）
            time.sleep(0.008)


# ===================== 程序入口 =====================
if __name__ == "__main__":
    try:
        # 启动主函数
        main()
    except KeyboardInterrupt:
        # 捕获用户手动中断（Ctrl+C），友好退出
        print("\n✅ 程序被用户手动中断，正常退出")
    except Exception as e:
        # 捕获其他异常，打印错误信息
        print(f"\n❌ 程序运行异常：{e}")