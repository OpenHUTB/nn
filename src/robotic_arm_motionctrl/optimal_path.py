import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.optimize import linear_sum_assignment

# ===================== 模型 =====================
XML_PATH = "arm6dof_final.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
JOINT_NUM = model.nu
DT = 0.005

# ===================== 随机巡检点位 =====================
points = np.array([
    [0.5,  0.0, 0.4],
    [0.4,  0.3, 0.4],
    [0.4, -0.3, 0.4],
    [-0.4, 0.3, 0.4],
    [-0.4, -0.3, 0.4],
    [0.0,  0.4, 0.4],
    [0.0, -0.4, 0.4],
])

# 扩展成关节目标（前3轴动，后面不动）
def to_q(p):
    return np.concatenate([p, [0,0,0]] + [np.zeros(JOINT_NUM-6)])

waypoints = [to_q(p) for p in points]

# ===================== TSP 最优路径规划 =====================
def tsp_order(points):
    n = len(points)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i,j] = np.linalg.norm(points[i]-points[j])
    _, idx = linear_sum_assignment(cost)
    return idx

order = tsp_order(points)
optimal_way = [waypoints[i] for i in order]
print("最优巡检顺序:", order + 1)
print("按最短路径自动巡检")

# ===================== PID =====================
class PID:
    def __init__(self):
        self.kp = 65
        self.ki = 1
        self.kd = 10
        self.last_err = np.zeros(JOINT_NUM)
        self.integ = np.zeros(JOINT_NUM)

    def ctrl(self, tar, cur):
        e = tar - cur
        self.integ = np.clip(self.integ + e*DT, -20,20)
        d = (e-self.last_err)/DT if DT>1e-6 else 0
        o = self.kp*e + self.ki*self.integ + self.kd*d
        self.last_err = e.copy()
        return np.clip(o, -160,160)

def interp(q0,q1,t,T):
    x = np.clip(t/T,0,1)
    s = 3*x*x-2*x**3
    return (1-s)*q0 + s*q1

# ===================== 主程序 =====================
def main():
    viewer = mujoco.viewer.launch_passive(model, data)
    pid = PID()
    idx = 0
    t = 0
    MOVE = 1.0
    WAIT = 0.3
    q_last = data.qpos[:JOINT_NUM].copy()

    print("开始 TSP 最优路径巡检")
    while viewer.is_running():
        q_cur = data.qpos[:JOINT_NUM].copy()
        q_tar = optimal_way[idx]

        if t < MOVE:
            q_des = interp(q_last, q_tar, t, MOVE)
        else:
            q_des = q_tar

        tau = pid.ctrl(q_des, q_cur)
        data.ctrl[:JOINT_NUM] = tau

        mujoco.mj_step(model, data)
        viewer.sync()
        t += DT
        time.sleep(DT)

        if t >= MOVE + WAIT:
            t = 0
            q_last = q_tar
            idx = (idx + 1) % len(optimal_way)
            print(f"→ 前往点位: {order[idx%len(order)]+1}")

    viewer.close()

if __name__ == "__main__":
    main()