import threading, queue
import tkinter as tk
import glfw, mujoco as mj
import numpy as np

XML_PATH = 'arm_with_gripper.xml'

CTRL_QUEUE = queue.Queue()        # 6 关节指令
GRIP_QUEUE = queue.Queue()        # 抓取状态 True=闭合 False=张开

# ---------- Tkinter 面板 ----------
def tk_panel(joint_names, joint_ranges):
    root = tk.Tk()
    root.title('Arm Panel + Grab')

    # 6 个关节滑杆
    scales, vars = [], [tk.DoubleVar() for _ in joint_names]
    def send_joint(*_):
        ctrl = np.clip([v.get() for v in vars],
                       [r[0] for r in joint_ranges],
                       [r[1] for r in joint_ranges])
        CTRL_QUEUE.put(ctrl)

    for i, (name, rng) in enumerate(zip(joint_names, joint_ranges)):
        tk.Label(root, text=name).grid(row=i, column=0, sticky='e')
        s = tk.Scale(root, from_=rng[0], to=rng[1], resolution=0.01,
                     orient=tk.HORIZONTAL, length=300, variable=vars[i],
                     command=send_joint)
        s.set((rng[0] + rng[1]) * 0.5)
        s.grid(row=i, column=1)
        scales.append(s)

    # Grab / Release 按钮
    def grab():
        GRIP_QUEUE.put(True)
    def release():
        GRIP_QUEUE.put(False)

    tk.Button(root, text='Grab', command=grab, bg='green', fg='white').grid(row=6, column=0, pady=10)
    tk.Button(root, text='Release', command=release, bg='red', fg='white').grid(row=6, column=1, pady=10)

    root.mainloop()

# ---------- MuJoCo 渲染线程 ----------
def mujoco_thread(joint_ids, joint_ranges, grip_act_ids):
    model = mj.MjModel.from_xml_path(XML_PATH)
    data  = mj.MjData(model)

    # 初始值
    ctrl_jnt = np.array([(r[0] + r[1]) * 0.5 for r in joint_ranges])
    ctrl_grip_open = np.array([0.05, 1.5, 1.2] * 3)   # 3 指张开（对应 slide/hinge1/hinge2）
    data.ctrl[joint_ids]      = ctrl_jnt
    data.ctrl[grip_act_ids]   = ctrl_grip_open
    CTRL_QUEUE.put(ctrl_jnt)
    GRIP_QUEUE.put(False)

    if not glfw.init():
        raise RuntimeError('glfw init failed')
    win = glfw.create_window(1200, 900, 'Arm Panel + Grab (Fixed)', None, None)
    if not win:
        glfw.terminate(); raise RuntimeError('glfw window failed')
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    cam = mj.MjvCamera()
    cam.azimuth, cam.elevation = 135, -20
    cam.distance = 1.8
    cam.lookat[:] = [0, 0, 0.3]
    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)
    scene = mj.MjvScene(model, maxgeom=5000)
    con = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)

    # 添加状态保持变量
    last_ctrl_jnt = ctrl_jnt.copy()
    last_grab_state = False
    
    # 添加标志位，用于跟踪是否已经处理过初始队列消息
    processed_initial_ctrl = False
    processed_initial_grip = False

    while not glfw.window_should_close(win):
        glfw.poll_events()
        width, height = glfw.get_framebuffer_size(win)
        viewport = mj.MjrRect(0, 0, width, height)

        # 1. 关节指令
        try:
            ctrl_jnt = CTRL_QUEUE.get_nowait()
            # 只有在处理过初始消息后才更新状态
            if processed_initial_ctrl:
                ctrl_jnt = np.clip(ctrl_jnt,
                                   [r[0] for r in joint_ranges],
                                   [r[1] for r in joint_ranges])
                data.ctrl[joint_ids] = ctrl_jnt
                last_ctrl_jnt = ctrl_jnt.copy()  # 保存最后一次控制指令
            else:
                processed_initial_ctrl = True
        except queue.Empty:
            # 如果没有新的控制指令，保持上次的状态
            data.ctrl[joint_ids] = last_ctrl_jnt

        # 2. 抓取指令（9 个执行器一起写，保证 equality 同步）
        try:
            grab = GRIP_QUEUE.get_nowait()
            # 只有在处理过初始消息后才更新状态
            if processed_initial_grip:
                if grab:          # 闭合
                    data.ctrl[grip_act_ids] = [-0.02, -0.2, 0.0] * 3
                else:             # 张开
                    data.ctrl[grip_act_ids] = [0.05, 1.5, 1.2] * 3
                last_grab_state = grab  # 保存最后一次抓取状态
            else:
                processed_initial_grip = True
        except queue.Empty:
            # 如果没有新的抓取指令，保持上次的状态
            if last_grab_state:  # 闭合状态
                data.ctrl[grip_act_ids] = [-0.02, -0.2, 0.0] * 3
            else:  # 张开状态
                data.ctrl[grip_act_ids] = [0.05, 1.5, 1.2] * 3

        mj.mj_step(model, data)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(viewport, scene, con)
        glfw.swap_buffers(win)

    glfw.terminate()

# ---------- 主入口 ----------
def main():
    model = mj.MjModel.from_xml_path(XML_PATH)

    # 6 个 arm 执行器
    jnt_act = ['motor_joint1', 'motor_joint2', 'motor_joint3',
               'motor_joint4', 'motor_joint5', 'motor_joint6']
    # 9 个夹爪执行器（3 指 × 3 自由度）
    grip_act = ['motor_finger1_slide', 'motor_finger1_hinge1', 'motor_finger1_hinge2',
                'motor_finger2_slide', 'motor_finger2_hinge1', 'motor_finger2_hinge2',
                'motor_finger3_slide', 'motor_finger3_hinge1', 'motor_finger3_hinge2']

    joint_ids  = [model.actuator(n).id for n in jnt_act]
    grip_ids   = [model.actuator(n).id for n in grip_act]

    # 用关节真实 range 做硬限位
    joint_ranges = [model.jnt_range[model.joint(j.replace('motor_', '')).id] for j in jnt_act]

    threading.Thread(target=tk_panel, args=(jnt_act, joint_ranges), daemon=True).start()
    mujoco_thread(joint_ids, joint_ranges, grip_ids)

if __name__ == '__main__':
    main()