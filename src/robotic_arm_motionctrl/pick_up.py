import numpy as np
import mujoco
import mujoco.viewer
import time

# ===================== 模型配置（修正笔误） =====================
XML_PATH = "arm6dof_final.xml"  # 你的模型文件路径
model = mujoco.MjModel.from_xml_path(XML_PATH)  # 修正：XML → XML_PATH
data = mujoco.MjData(model)
JOINT_NUM = model.nu
DT = 0.005

# ===================== 标准拾放点位（动作直观） =====================
HOME      = np.zeros(JOINT_NUM)

# 抓取点（右前方）
PICK_UP    = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0] + [0]*(JOINT_NUM-6))  # 物体上方
PICK_DOWN  = np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0] + [0]*(JOINT_NUM-6))  # 下探抓取

# 放置点（左前方）
PLACE_UP   = np.array([-0.5, 0.0, 0.5, 0.0, 0.0, 0.0] + [0]*(JOINT_NUM-6)) # 目标上方
PLACE_DOWN = np.array([-0.5, 0.0, 0.2, 0.0, 0.0, 0.0] + [0]*(JOINT_NUM-6)) # 下放放置

# ===================== 状态机（流程清晰） =====================
STATES = [
    "HOME",
    "GO_PICK_UP",
    "GO_PICK_DOWN",
    "DELAY_GRAB",
    "BACK_PICK_UP",
    "GO_PLACE_UP",
    "GO_PLACE_DOWN",
    "DELAY_RELEASE",
    "BACK_PLACE_UP",
    "GO_HOME",
    "FINISH"
]

# ===================== PID控制器（跟点更准） =====================
class PID:
    def __init__(self):
        self.kp = 70
        self.ki = 1
        self.kd = 12
        self.last_err = np.zeros(JOINT_NUM)
        self.integ = np.zeros(JOINT_NUM)

    def ctrl(self, target, curr):
        err = target - curr
        self.integ += err * DT
        self.integ = np.clip(self.integ, -20, 20)  # 抗积分饱和
        d = (err - self.last_err)/DT if DT > 1e-6 else 0
        out = self.kp*err + self.ki*self.integ + self.kd*d
        out = np.clip(out, -150, 150)  # 输出限幅
        self.last_err = err.copy()
        return out

# 平滑轨迹插值（S曲线，无冲击）
def interp(q0, q1, t, T):
    x = np.clip(t/T, 0, 1)
    s = 3*x**2 - 2*x**3  # S曲线插值
    return q0*(1-s) + q1*s

# ===================== 主程序 =====================
def main():
    # 启动仿真可视化
    viewer = mujoco.viewer.launch_passive(model, data)
    pid = PID()

    # 状态初始化
    state = 0
    t = 0.0
    MOVE_T = 1.2  # 每个移动阶段的时间（秒）
    WAIT_T = 0.5  # 抓取/放置停顿时间
    q_last = data.qpos[:JOINT_NUM].copy()

    print("=== 标准拾放作业开始 ===")
    print("动作流程：原点→抓取点上方→下探→抓取→抬升→放置点上方→下放→放置→抬升→回原点")

    while viewer.is_running():
        q_curr = data.qpos[:JOINT_NUM].copy()
        now_state = STATES[state]

        # --- 核心状态逻辑（一步一动作，直观） ---
        if now_state == "HOME":
            q_des = HOME
            if np.linalg.norm(q_curr - q_des) < 0.01:  # 到达原点
                state = 1
                q_last = q_curr
                t = 0.0
                print("→ 前往抓取点上方")

        elif now_state == "GO_PICK_UP":
            q_des = interp(q_last, PICK_UP, t, MOVE_T)
            if t >= MOVE_T:  # 到达抓取点上方
                state = 2
                q_last = q_curr
                t = 0.0
                print("→ 下探抓取")

        elif now_state == "GO_PICK_DOWN":
            q_des = interp(q_last, PICK_DOWN, t, MOVE_T)
            if t >= MOVE_T:  # 到达抓取位置
                state = 3
                t = 0.0
                print("→ 模拟抓取（停顿）")

        elif now_state == "DELAY_GRAB":
            q_des = PICK_DOWN
            if t >= WAIT_T:  # 抓取停顿完成
                state = 4
                q_last = q_curr
                t = 0.0
                print("→ 抓取完成，抬升")

        elif now_state == "BACK_PICK_UP":
            q_des = interp(q_last, PICK_UP, t, MOVE_T)
            if t >= MOVE_T:  # 抬升完成
                state = 5
                q_last = q_curr
                t = 0.0
                print("→ 前往放置点上方")

        elif now_state == "GO_PLACE_UP":
            q_des = interp(q_last, PLACE_UP, t, MOVE_T)
            if t >= MOVE_T:  # 到达放置点上方
                state = 6
                q_last = q_curr
                t = 0.0
                print("→ 下放放置")

        elif now_state == "GO_PLACE_DOWN":
            q_des = interp(q_last, PLACE_DOWN, t, MOVE_T)
            if t >= MOVE_T:  # 到达放置位置
                state = 7
                t = 0.0
                print("→ 模拟放置（停顿）")

        elif now_state == "DELAY_RELEASE":
            q_des = PLACE_DOWN
            if t >= WAIT_T:  # 放置停顿完成
                state = 8
                q_last = q_curr
                t = 0.0
                print("→ 放置完成，抬升")

        elif now_state == "BACK_PLACE_UP":
            q_des = interp(q_last, PLACE_UP, t, MOVE_T)
            if t >= MOVE_T:  # 抬升完成
                state = 9
                q_last = q_curr
                t = 0.0
                print("→ 返回原点")

        elif now_state == "GO_HOME":
            q_des = interp(q_last, HOME, t, MOVE_T*1.2)
            if t >= MOVE_T*1.2:  # 回到原点
                state = 10
                print("✅ 拾放作业完成！")

        # PID控制输出
        tau = pid.ctrl(q_des, q_curr)
        data.ctrl[:JOINT_NUM] = tau

        # 仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()

        # 时间更新
        t += DT
        time.sleep(DT)

    viewer.close()

if __name__ == "__main__":
    main()