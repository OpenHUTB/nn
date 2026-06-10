"""
机械臂运动学优化模块
=================================

性能优化点：
1. 轨迹规划 - Numba加速 + 并行计算
2. 解析雅可比 - 替代数值微分
3. 逆运动学 - 解析解 + 迭代混合
4. 碰撞检测 - 空间哈希 + SAP算法

使用示例：
    from core.optimized_kinematics import OptimizedKinematics
    
    kin = OptimizedKinematics()
    
    # 正运动学
    pose = kin.forward_kinematics([30, 45, -30, 60, 0, 90])
    
    # 轨迹规划
    traj_pos, traj_vel = kin.plan_trajectory(start, target)
    
    # 逆运动学
    joints = kin.inverse_kinematics([0.3, 0.1, 0.4])
    
    # 碰撞检测
    collision, pairs = kin.check_collision(joints, obstacles)

作者: 机械臂控制团队
版本: 1.0.0
"""

import numpy as np
import math
import yaml
import logging
from numba import njit, prange, float64, int32
from functools import lru_cache
from typing import Tuple, Optional, List

# ============================================================
# 异常类定义
# ============================================================

class KinematicsError(Exception):
    """运动学基础异常类"""
    pass

class ConfigLoadError(KinematicsError):
    """配置加载异常"""
    pass

class JointLimitError(KinematicsError):
    """关节限位异常"""
    pass

class IKSolverError(KinematicsError):
    """逆运动学求解异常"""
    pass

class TrajectoryError(KinematicsError):
    """轨迹规划异常"""
    pass

# ============================================================
# 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Numba加速的轨迹规划函数
# ============================================================

@njit(cache=True)
def _plan_trapezoid_numba(delta: float, max_vel: float, max_acc: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba加速的梯形轨迹规划（单关节）
    
    梯形速度曲线由三段组成：加速段 -> 匀速段 -> 减速段
    
    参数:
        delta: 关节角度变化量（弧度）
        max_vel: 最大速度 (rad/s)
        max_acc: 最大加速度 (rad/s^2)
        dt: 时间步长 (s)
    
    返回:
        (位置数组, 速度数组)
    
    算法原理:
        1. 计算加速段距离 accel_dist = v^2 / (2*a)
        2. 根据总距离判断是否需要匀速段
        3. 沿时间轴积分得到位置和速度
    """
    # 小位移处理：避免数值不稳定
    if abs(delta) < 1e-8:
        return np.array([0.0]), np.array([0.0])
    
    direction = 1.0 if delta > 0 else -1.0
    dist = abs(delta)
    # 加速/减速阶段所需的距离
    accel_dist = (max_vel ** 2) / (2 * max_acc)
    
    # 计算各段时间
    if dist <= 2 * accel_dist:
        # 距离不足，无需匀速段（三角形速度曲线）
        peak_vel = math.sqrt(dist * max_acc)
        accel_time = peak_vel / max_acc
        uniform_time = 0.0
    else:
        # 标准梯形曲线
        accel_time = max_vel / max_acc
        uniform_time = (dist - 2 * accel_dist) / max_vel
    
    total_time = 2 * accel_time + uniform_time
    num_steps = max(2, int(total_time / dt) + 1)
    dt_actual = total_time / (num_steps - 1)
    
    # 预分配数组（避免循环内分配）
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    
    # 时间轴积分
    for k in range(num_steps):
        t = k * dt_actual
        
        if t <= accel_time:
            # ========== 加速段 ==========
            # v(t) = a * t, s(t) = 0.5 * a * t^2
            velocities[k] = max_vel * t / accel_time * direction
            positions[k] = 0.5 * max_acc * t * t * direction
        elif t <= accel_time + uniform_time:
            # ========== 匀速段 ==========
            # v(t) = const, s(t) = v * t
            t_dec = t - accel_time
            velocities[k] = max_vel * direction
            positions[k] = (accel_dist + max_vel * t_dec) * direction
        else:
            # ========== 减速段 ==========
            # v(t) = v0 - a*t, s(t) = s0 + v0*t - 0.5*a*t^2
            t_dec = t - accel_time - uniform_time
            remaining = dist - accel_dist
            velocities[k] = (max_vel - max_acc * t_dec) * direction
            positions[k] = (remaining + max_vel * t_dec - 0.5 * max_acc * t_dec * t_dec) * direction
    
    # 确保端点精确
    positions[-1] = delta
    velocities[-1] = 0.0
    
    return positions, velocities


@njit(cache=True, parallel=True)
def _plan_all_joints_numba(start: np.ndarray, target: np.ndarray, 
                          max_vel: np.ndarray, max_acc: np.ndarray, 
                          dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba并行规划所有关节的梯形轨迹
    
    使用 prange 实现多核并行，每个关节独立规划
    
    参数:
        start: 起始关节角 (弧度), shape: (n,)
        target: 目标关节角 (弧度), shape: (n,)
        max_vel: 各关节最大速度 (rad/s), shape: (n,)
        max_acc: 各关节最大加速度 (rad/s^2), shape: (n,)
        dt: 时间步长 (s)
    
    返回:
        (轨迹位置, 轨迹速度), shape: (N, n), N为轨迹点数
    """
    n_joints = len(start)
    max_points = 2000  # 最大轨迹点数（避免动态分配）
    
    # 预分配结果数组（Numba要求固定形状）
    traj_pos = np.zeros((max_points, n_joints))
    traj_vel = np.zeros((max_points, n_joints))
    
    # ========== 并行规划每个关节 ==========
    # prange 开启多线程，迭代之间无依赖
    for i in prange(n_joints):
        # 单关节梯形规划
        pos, vel = _plan_trapezoid_numba(target[i] - start[i], max_vel[i], max_acc[i], dt)
        n_pts = len(pos)
        # 填充到结果数组
        for j in range(n_pts):
            traj_pos[j, i] = start[i] + pos[j]
            traj_vel[j, i] = vel[j]
    
    # ========== 计算实际轨迹长度 ==========
    # 找出最长关节的规划点数作为轨迹长度
    max_len = 0
    for i in range(n_joints):
        delta = abs(target[i] - start[i])
        if delta > 1e-8:
            accel_dist = (max_vel[i] ** 2) / (2 * max_acc[i])
            if delta <= 2 * accel_dist:
                peak_vel = math.sqrt(delta * max_acc[i])
                total_time = 2 * peak_vel / max_acc[i]
            else:
                total_time = 2 * max_vel[i] / max_acc[i] + (delta - 2 * accel_dist) / max_vel[i]
            max_len = max(max_len, int(total_time / dt) + 1)
    
    return traj_pos[:max_len], traj_vel[:max_len]


# ============================================================
# 解析雅可比计算类
# ============================================================

class AnalyticJacobian:
    """
    六轴机械臂解析雅可比计算器
    
    相比数值微分，解析雅可比具有以下优势：
    1. 无精度损失（数值微分存在截断误差）
    2. 无需多次调用正运动学（数值微分需要2n次FK）
    3. 可计算任意点的精确雅可比
    
    数学原理：
        位置雅可比: Jp_i = z_{i-1} × (p_ee - p_{i-1})
        其中 z_{i-1} 是关节i的旋转轴方向
              p_ee 是末端执行器位置
              p_{i-1} 是关节i的位置
    """
    
    def __init__(self, dh_fixed: dict):
        """
        初始化解析雅可比计算器
        
        参数:
            dh_fixed: D-H参数字典，格式 {关节索引: {'a': 0.2, 'alpha': 90, 'd': 0.1, 'theta': 0}}
        """
        self.dh_fixed = dh_fixed
        self.n = len(dh_fixed)
        self._precompute_chain_params()
    
    def _precompute_chain_params(self):
        """预计算链式参数"""
        # 从D-H参数提取各关节的几何参数
        self.link_lengths = np.array([p['a'] for p in self.dh_fixed.values()])  # 连杆长度 a
        self.joint_offsets = np.array([p['d'] for p in self.dh_fixed.values()])  # 关节偏距 d
        self.twist_angles = np.array([p['alpha_rad'] for p in self.dh_fixed.values()])  # 扭转角 α
    
    def compute(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """
        计算解析雅可比矩阵（位置部分）
        
        参数:
            joint_angles_rad: 关节角度（弧度），shape: (n,)
        
        返回:
            J: 位置雅可比矩阵，shape: (3, n)
        
        算法步骤：
            1. 依次计算各关节的变换矩阵 T_i
            2. 记录各关节的位置 p_i 和旋转轴 z_i
            3. 对每个关节计算 Jp_i = z_{i-1} × (p_ee - p_{i-1})
        """
        J = np.zeros((3, self.n), dtype=np.float64)
        
        # 计算各关节位置和旋转轴方向
        positions = []
        z_axes = []
        T = np.eye(4, dtype=np.float64)
        
        for i in range(self.n):
            # 提取第i关节的参数
            a = self.link_lengths[i]       # 连杆长度
            alpha = self.twist_angles[i]   # 扭转角
            d = self.joint_offsets[i]      # 关节偏距
            theta = joint_angles_rad[i]    # 关节转角
            
            # ========== D-H 标准变换矩阵 ==========
            # T_i = Rz(theta) * Tz(d) * Rx(alpha) * Rx(a)
            ca, sa = math.cos(alpha), math.sin(alpha)
            ct, st = math.cos(theta), math.sin(theta)
            
            Ti = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            
            # 累积变换
            T = T @ Ti
            # 记录关节位置和旋转轴
            positions.append(T[:3, 3].copy())
            z_axes.append(T[:3, 2].copy())  # z轴是旋转轴
        
        # 末端位置
        ee_pos = positions[-1]
        
        # ========== 计算雅可比列 ==========
        for i in range(self.n):
            # 关节i的雅可比列
            # 第一个关节：z0 = [0,0,1]（默认世界z轴）
            z_prev = z_axes[i-1] if i > 0 else np.array([0., 0., 1.])
            # 位置差：末端到关节i的位置
            p_diff = ee_pos - positions[i-1] if i > 0 else ee_pos
            # 叉积得到线速度分量
            J[:, i] = np.cross(z_prev, p_diff)
        
        return J
    
    def compute_full(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """
        计算完整6x6雅可比（位置 + 姿态）
        
        参数:
            joint_angles_rad: 关节角度（弧度）
        
        返回:
            完整雅可比矩阵，shape: (6, n)
        """
        J_pos = self.compute(joint_angles_rad)
        J_ori = self._compute_orientation_jacobian(joint_angles_rad)
        return np.vstack([J_pos, J_ori])
    
    def _compute_orientation_jacobian(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """
        计算姿态雅可比（旋转矩阵的导数）
        
        姿态雅可比：关节i的旋转轴方向 z_{i-1}
        """
        J_ori = np.zeros((3, self.n), dtype=np.float64)
        
        z_prev = np.array([0., 0., 1.])  # 基座默认z轴向上
        for i in range(self.n):
            J_ori[:, i] = z_prev
            if i < self.n - 1:
                # 累积旋转：沿链传递
                alpha = self.twist_angles[i]
                ca, sa = math.cos(alpha), math.sin(alpha)
                theta = joint_angles_rad[i]
                ct, st = math.cos(theta), math.sin(theta)
                # 计算下一个关节的旋转轴
                z_prev = np.array([-st*sa, ct*sa, ca])
        
        return J_ori


# ============================================================
# 混合逆运动学求解器
# ============================================================

class HybridIKSolver:
    """
    混合逆运动学求解器
    
    结合解析闭式解和迭代优化的优势：
    1. 前3个关节（肩部+肘部）：使用解析几何求解，无需迭代
    2. 后3个关节（腕部）：使用阻尼最小二乘迭代优化
    
    适用构型：球形手腕机械臂（常见6轴工业机器人）
    
    数学原理：
        - 解析解：基于平面几何和余弦定理
        - 迭代解：阻尼最小二乘 J^T * error / (J*J^T + λI)
    """
    
    def __init__(self, dh_fixed: dict):
        """
        初始化混合求解器
        
        参数:
            dh_fixed: D-H参数字典
        """
        self.dh_fixed = dh_fixed
        self.n = len(dh_fixed)
        self._extract_geometric_params()
    
    def _extract_geometric_params(self):
        """提取几何参数用于解析求解"""
        # 大臂长度（前两个关节间的距离）
        self.l1 = max(self.dh_fixed[1]['a'] / 1000 + self.dh_fixed[2]['a'] / 1000, 0.01)
        # 小臂长度（第3-4关节间的距离）
        self.l2 = max(self.dh_fixed[3]['a'] / 1000 + self.dh_fixed[4]['a'] / 1000, 0.01)
        # 腕部偏移
        self.l3 = self.dh_fixed[5]['d'] / 1000
        # 基座高度偏移
        self.h = self.dh_fixed[0]['d'] / 1000 + self.dh_fixed[4]['d'] / 1000
    
    def solve(self, target_pos: np.ndarray, initial_joints: np.ndarray, 
              tolerance: float = 1e-4, max_iter: int = 50) -> np.ndarray:
        """
        混合逆运动学求解
        
        参数:
            target_pos: 目标位置 [x, y, z] (米)
            initial_joints: 初始关节角（度）
            tolerance: 位置误差容限（米），默认0.1mm
            max_iter: 最大迭代次数，默认50
        
        返回:
            关节角数组（度）
        
        异常:
            IKSolverError: 目标超出工作空间或迭代不收敛
        """
        target_pos = np.array(target_pos, dtype=np.float64)
        
        # 验证目标位置是否在可达范围内
        max_reach = self.l1 + self.l2
        distance = np.linalg.norm(target_pos[:2])  # 水平距离
        height = target_pos[2] - self.h  # 相对高度
        if np.sqrt(distance**2 + height**2) > max_reach * 0.98:
            logger.warning(f"目标位置接近工作空间边界: {distance:.3f}m")
        
        # ========== 步骤1: 解析求解前3个关节 ==========
        j1, j2, j3 = self._solve_spherical_wrist(target_pos)
        
        # ========== 步骤2: 初始化关节角 ==========
        current = np.array(initial_joints, dtype=np.float64) * np.pi / 180.0
        if j1 is not None:
            current[0] = j1
        if j2 is not None:
            current[1] = j2
        if j3 is not None:
            current[2] = j3
        
        # ========== 步骤3: 迭代优化所有关节 ==========
        current = self._iterative_refinement(current, target_pos, tolerance, max_iter)
        
        return current * 180.0 / np.pi
    
    def _solve_spherical_wrist(self, target: np.ndarray) -> Tuple[float, float, float]:
        """
        解析求解球形手腕机械臂的前3个关节
        
        算法：
            1. 关节1：绕z轴旋转，使机械臂朝向目标
            2. 关节2-3：使用余弦定理求解平面几何
        
        参数:
            target: 目标位置 [x, y, z]
        
        返回:
            (j1, j2, j3) 弧度，若解不存在则返回 None
        """
        x, y, z = target[0], target[1], target[2]
        
        # ========== 关节1: 基座旋转 ==========
        # atan2(y, x) 得到从x轴正方向逆时针旋转的角度
        if abs(x) > 1e-6 or abs(y) > 1e-6:
            j1 = math.atan2(y, x)
        else:
            j1 = 0.0
        
        # ========== 计算在水平面上的投影距离 ==========
        r = math.sqrt(x**2 + y**2)  # 水平距离
        h = z - self.h  # 相对高度
        
        # ========== 求解2-3关节（平面问题） ==========
        # 使用余弦定理: c^2 = a^2 + b^2 - 2ab*cos(C)
        # 其中 D 是肘部到腕部的距离与目标点的比值
        D = (r**2 + h**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        
        # D > 1 表示目标超出工作空间
        if abs(D) > 1.0:
            logger.warning(f"目标超出工作空间: D={D:.3f}")
            return j1, None, None
        
        # 肘部角度（负号表示肘下构型）
        elbow_angle = math.atan2(-math.sqrt(1 - D**2), D)
        
        # 肩部角度
        alpha = math.atan2(h, r)  # 目标方向角
        beta = math.atan2(self.l2 * math.sin(elbow_angle), 
                         self.l1 + self.l2 * math.cos(elbow_angle))
        j2 = alpha - beta
        
        # 关节3：使腕部朝向一致（简化处理）
        j3 = -j2 - elbow_angle
        
        return j1, j2, j3
    
    def _iterative_refinement(self, joints: np.ndarray, target: np.ndarray,
                              tol: float, max_iter: int) -> np.ndarray:
        """
        阻尼最小二乘法迭代优化
        
        使用阻尼最小二乘法（Damped Least Squares）迭代求解逆运动学
        相比纯伪逆法，阻尼项提高了数值稳定性
        
        参数:
            joints: 当前关节角（弧度）
            target: 目标位置
            tol: 收敛容差
            max_iter: 最大迭代次数
        
        返回:
            优化后的关节角（弧度）
        """
        jacobian = AnalyticJacobian(self.dh_fixed)
        
        for iteration in range(max_iter):
            # 计算当前末端位置
            T = self._compute_fk(joints)
            current_pos = T[:3, 3]
            
            # 计算位置误差
            error = target - current_pos
            error_norm = np.linalg.norm(error)
            
            # 收敛检查
            if error_norm < tol:
                logger.debug(f"IK收敛: 迭代{iteration}次, 误差{error_norm:.6f}m")
                break
            
            # 计算雅可比矩阵
            J = jacobian.compute(joints)
            
            # ========== 阻尼最小二乘求解 ==========
            # delta = J^T * error / (J*J^T + λ²*I)
            # λ 是阻尼因子，较大时收敛慢但稳定，较小时快但可能振荡
            lamda = 0.01
            JtJ = J @ J.T + lamda * np.eye(3)
            delta = J.T @ np.linalg.solve(JtJ, error)
            
            # 更新关节角（步长限制避免震荡）
            joints = joints + delta * 0.5
            joints = np.clip(joints, -np.pi, np.pi)
        
        else:
            # 迭代未收敛警告
            logger.warning(f"IK迭代{max_iter}次未收敛, 最终误差{error_norm:.6f}m")
        
        return joints
    
    def _compute_fk(self, joints: np.ndarray) -> np.ndarray:
        """
        计算正运动学
        
        参数:
            joints: 关节角（弧度）
        
        返回:
            末端执行器齐次变换矩阵 4x4
        """
        T = np.eye(4, dtype=np.float64)
        
        for i in range(len(self.dh_fixed)):
            params = self.dh_fixed[i]
            a = params['a']
            alpha = params['alpha_rad']
            d = params['d']
            theta_base = params['theta_base']
            theta = joints[i] + theta_base * np.pi / 180.0
            
            ca, sa = math.cos(alpha), math.sin(alpha)
            ct, st = math.cos(theta), math.sin(theta)
            
            Ti = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            T = T @ Ti
        
        return T


# ============================================================
# 空间哈希碰撞检测
# ============================================================

class SpatialHashCollision:
    """
    空间哈希碰撞检测器
    
    使用均匀网格空间哈希实现O(n)平均复杂度的碰撞检测，
    相比O(n²)的穷举检测，在稀疏场景下效率提升明显
    
    工作原理：
        1. 将3D空间划分为固定大小的网格单元
        2. 每个物体插入到其占据的所有网格单元
        3. 检测时只查询物体所在单元及其邻近单元
        
    适用场景：物体分布稀疏、碰撞体数量较多的场景
    """
    
    def __init__(self, cell_size: float = 0.05):
        """
        参数:
            cell_size: 网格单元大小（米）
                       通常设为最大碰撞体尺寸的2倍
                       过大：查询效率降低
                       过小：物体跨多个单元，插入开销增加
        """
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size  # 预计算除法
        self._grid = {}  # 空间哈希表 {(ix,iy,iz): [(obj_id, pos, radius), ...]}
    
    def _hash_pos(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """
        计算位置的哈希键
        
        参数:
            pos: 位置坐标 (x, y, z)
        
        返回:
            网格单元索引 (ix, iy, iz)
        """
        ix = int(math.floor(pos[0] * self.inv_cell_size))
        iy = int(math.floor(pos[1] * self.inv_cell_size))
        iz = int(math.floor(pos[2] * self.inv_cell_size))
        return (ix, iy, iz)
    
    def clear(self):
        """清空哈希表"""
        self._grid.clear()
    
    def insert(self, obj_id: int, pos: np.ndarray, radius: float):
        """
        插入碰撞体到空间哈希
        
        球体可能占据多个网格单元（取决于半径和单元大小）
        """
        # 计算球体占据的网格范围
        r = radius * self.inv_cell_size
        base = self._hash_pos(pos)
        
        # 遍历球体占据的所有网格单元
        for dx in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
            for dy in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
                for dz in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
                    key = (base[0] + dx, base[1] + dy, base[2] + dz)
                    if key not in self._grid:
                        self._grid[key] = []
                    self._grid[key].append((obj_id, pos, radius))
    
    def query_nearby(self, pos: np.ndarray, radius: float) -> List[Tuple]:
        """
        查询可能与给定位置发生碰撞的物体
        
        参数:
            pos: 查询位置
            radius: 查询物体的半径
        
        返回:
            可能碰撞的物体列表 [(obj_id, pos, radius), ...]
        """
        nearby = []
        r = radius * self.inv_cell_size
        center = self._hash_pos(pos)
        
        # 查询中心单元及其邻近单元
        for dx in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
            for dy in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
                for dz in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
                    key = (center[0] + dx, center[1] + dy, center[2] + dz)
                    if key in self._grid:
                        nearby.extend(self._grid[key])
        
        return nearby
    
    def check_collision_pairs(self, positions: np.ndarray, radii: np.ndarray) -> List[Tuple[int, int]]:
        """
        检测所有碰撞对（O(n)平均复杂度）
        
        参数:
            positions: 物体位置数组 shape: (n, 3)
            radii: 物体半径数组 shape: (n,)
        
        返回:
            碰撞对列表 [(i, j), ...]，i < j
        """
        self.clear()
        n = len(positions)
        
        # ========== 步骤1: 构建空间哈希 ==========
        for i in range(n):
            self.insert(i, positions[i], radii[i])
        
        # ========== 步骤2: 检测碰撞 ==========
        checked = set()  # 已检测的对，避免重复
        collisions = []
        
        for i in range(n):
            nearby = self.query_nearby(positions[i], radii[i])
            
            for j, pos_j, r_j in nearby:
                # 避免重复检测：只检测 i < j 的对
                if i >= j:
                    continue
                
                pair_key = (i, j)
                if pair_key in checked:
                    continue
                checked.add(pair_key)
                
                # ========== 精确碰撞检测 ==========
                # 球-球碰撞：距离 < 半径和
                dist = np.linalg.norm(positions[i] - pos_j)
                min_dist = radii[i] + r_j
                
                if dist < min_dist:
                    collisions.append((i, j))
        
        return collisions


# ============================================================
# SAP碰撞检测（扫描轴算法）
# ============================================================

class SAPCollision:
    """
    SAP (Sweep and Prune) 碰撞检测算法
    
    利用AABB（轴对齐包围盒）和一维排序实现高效的碰撞检测
    
    工作原理：
        1. 按某轴（如x轴）对所有AABB排序
        2. 沿轴线扫描，维护"活跃"AABB列表
        3. 只对在x轴上重叠的AABB进行完整检测
        
    适用场景：刚体数量较多、分布密集的场景
    
    复杂度：
        - 理想情况：O(n)
        - 最坏情况：O(n²)
    """
    
    def __init__(self):
        """初始化SAP检测器"""
        self._aabbs = []  # AABB包围盒列表
        self._active = []  # 活跃物体列表
    
    def update_aabb(self, obj_id: int, pos: np.ndarray, radius: float):
        """
        更新物体AABB（轴对齐包围盒）
        
        参数:
            obj_id: 物体ID
            pos: 物体位置
            radius: 物体半径
        """
        self._aabbs.append({
            'id': obj_id,
            'min': pos - radius,  # AABB最小点
            'max': pos + radius,  # AABB最大点
            'radius': radius,
            'pos': pos
        })
    
    def clear(self):
        """清空检测器"""
        self._aabbs.clear()
        self._active.clear()
    
    def check_collisions(self) -> List[Tuple[int, int]]:
        """
        执行SAP碰撞检测
        
        返回:
            碰撞对列表 [(id1, id2), ...]
        """
        if len(self._aabbs) < 2:
            return []
        
        # ========== 按x轴排序 ==========
        sorted_aabbs = sorted(self._aabbs, key=lambda a: (a['min'][0], a['id']))
        
        collisions = []
        active = []  # 当前活跃的AABB
        
        for aabb in sorted_aabbs:
            aabb_min = aabb['min']
            aabb_max = aabb['max']
            
            # ========== 移除不在范围内的活跃物体 ==========
            # 如果活跃物体的max < 当前AABB的min，则不可能再与后续物体碰撞
            new_active = []
            for active_aabb in active:
                if active_aabb['max'][0] < aabb_min[0]:
                    continue  # 移除
                new_active.append(active_aabb)
            active = new_active
            
            # ========== 与活跃物体检测碰撞 ==========
            for active_aabb in active:
                if self._aabb_overlap(active_aabb, aabb):
                    # AABB重叠，进行精确球-球碰撞检测
                    dist = np.linalg.norm(active_aabb['pos'] - aabb['pos'])
                    min_dist = active_aabb['radius'] + aabb['radius']
                    if dist < min_dist:
                        collisions.append((active_aabb['id'], aabb['id']))
            
            # 将当前AABB加入活跃列表
            active.append(aabb)
        
        return collisions
    
    def _aabb_overlap(self, a: dict, b: dict) -> bool:
        """
        检测两个AABB是否在三维空间重叠
        
        参数:
            a, b: AABB字典，包含'min'和'max'点
        
        返回:
            True 如果重叠
        """
        # AABB重叠条件：各轴投影都重叠
        return (a['min'][0] <= b['max'][0] and a['max'][0] >= b['min'][0] and
                a['min'][1] <= b['max'][1] and a['max'][1] >= b['min'][1] and
                a['min'][2] <= b['max'][2] and a['max'][2] >= b['min'][2])


# ============================================================
# 优化的运动学主类（统一接口）
# ============================================================

class OptimizedKinematics:
    """
    优化版机械臂运动学解算器
    
    整合了以下优化功能：
    1. 正运动学：缓存加速（LRU）
    2. 雅可比计算：解析法（替代数值微分）
    3. 逆运动学：混合求解（解析+迭代）
    4. 轨迹规划：Numba加速
    5. 碰撞检测：空间哈希/SAP算法
    
    使用示例：
        kin = OptimizedKinematics()
        
        # 正运动学
        pose = kin.forward_kinematics([30, 45, -30, 60, 0, 90])
        
        # 雅可比矩阵
        J = kin.compute_jacobian([30, 45, -30, 60, 0, 90])
        
        # 逆运动学
        joints = kin.inverse_kinematics([0.3, 0.1, 0.4])
        
        # 轨迹规划
        traj_pos, traj_vel = kin.plan_trajectory(start, target)
        
        # 碰撞检测
        collision, pairs = kin.check_collision(joints, obstacles)
    """
    
    def __init__(self, config_path: str = "config/arm_config.yaml"):
        """
        初始化优化运动学求解器
        
        参数:
            config_path: D-H配置文件路径
        
        异常:
            ConfigLoadError: 配置文件加载失败
        """
        self.config_path = config_path
        self.dh_params = {}
        self.joint_limits = {}
        self.joint_num = 6
        
        # ========== LRU缓存（避免重复计算） ==========
        self._fk_cache = {}         # 正运动学缓存
        self._jacobian_cache = {}   # 雅可比缓存
        self._cache_size = 1000     # 最大缓存条目
        
        # 加载配置
        self._load_config()
        self._precompute()
        
        # 初始化优化组件
        self.analytic_jacobian = AnalyticJacobian(self.dh_fixed)
        self.hybrid_ik = HybridIKSolver(self.dh_fixed)
        self.spatial_hash = SpatialHashCollision(cell_size=0.1)
        self.sap_collision = SAPCollision()
        
        # 轨迹规划参数（默认6轴参数）
        self.max_vel = np.array([60, 45, 45, 90, 90, 120]) * np.pi / 180.0  # rad/s
        self.max_acc = np.array([120, 90, 90, 180, 180, 240]) * np.pi / 180.0  # rad/s^2
        self.dt = 0.001  # 1ms控制周期
    
    def _load_config(self):
        """
        加载D-H参数配置文件
        
        异常:
            ConfigLoadError: 文件不存在或格式错误
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证必需字段
            if 'DH_PARAMS' not in config:
                raise ConfigLoadError(f"配置文件缺少 'DH_PARAMS' 字段: {self.config_path}")
            if 'JOINT_LIMITS' not in config:
                raise ConfigLoadError(f"配置文件缺少 'JOINT_LIMITS' 字段: {self.config_path}")
            
            self.dh_params = config['DH_PARAMS']
            self.joint_limits = config['JOINT_LIMITS']
            logger.info(f"成功加载配置: {self.config_path}")
            
        except FileNotFoundError:
            raise ConfigLoadError(f"配置文件不存在: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML解析失败: {e}")
    
    def _precompute(self):
        """
        预计算D-H参数的固定部分
        
        将mm转换为m，度转换为弧度
        """
        self.dh_fixed = {}
        for joint, params in self.dh_params.items():
            idx = int(joint.replace('joint', '')) - 1
            self.dh_fixed[idx] = {
                'a': params['a'] / 1000,  # mm -> m
                'alpha_rad': math.radians(params['alpha']),  # 度 -> 弧度
                'd': params['d'] / 1000,  # mm -> m
                'theta_base': params['theta']
            }
    
    def forward_kinematics(self, joint_angles_deg: np.ndarray, 
                          return_joints: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        优化的正运动学（缓存加速）
        
        参数:
            joint_angles_deg: 关节角度（度），shape: (n,)
            return_joints: 是否返回中间关节位置
        
        返回:
            末端位姿 [x, y, z, rx, ry, rz]（米，度）
            若 return_joints=True，还返回中间关节位置
        
        异常:
            ValueError: 关节角度数量不匹配
        """
        # ========== 参数验证 ==========
        joint_angles_deg = np.array(joint_angles_deg)
        if len(joint_angles_deg) != self.joint_num:
            raise ValueError(
                f"关节数不匹配: 需要{self.joint_num}个关节，"
                f"实际输入{len(joint_angles_deg)}个"
            )
        
        # ========== 缓存查询 ==========
        cache_key = tuple(np.round(joint_angles_deg, 1))
        if cache_key in self._fk_cache:
            cached = self._fk_cache[cache_key]
            if return_joints:
                return cached[:6], cached[6:]
            return cached[:6]
        
        # ========== 计算变换矩阵 ==========
        joints_rad = joint_angles_deg * np.pi / 180.0
        T = np.eye(4, dtype=np.float64)
        positions = []
        
        for i in range(self.joint_num):
            params = self.dh_fixed[i]
            a, alpha = params['a'], params['alpha_rad']
            d = params['d']
            theta = params['theta_base'] * np.pi / 180.0 + joints_rad[i]
            
            ca, sa = math.cos(alpha), math.sin(alpha)
            ct, st = math.cos(theta), math.sin(theta)
            
            T = T @ np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            positions.append(T[:3, 3].copy())
        
        # ========== 提取欧拉角 ==========
        pos = T[:3, 3]
        r31, r32, r33 = T[2, 0], T[2, 1], T[2, 2]
        r21, r11 = T[1, 0], T[0, 0]
        
        euler = np.array([
            math.degrees(math.atan2(r32, r33)),  # rx
            math.degrees(math.atan2(-r31, math.hypot(r32, r33))),  # ry
            math.degrees(math.atan2(r21, r11))  # rz
        ])
        
        result = np.concatenate([pos, euler])
        positions_arr = np.array(positions)
        
        # ========== 缓存管理（LUR淘汰） ==========
        if len(self._fk_cache) >= self._cache_size:
            self._fk_cache.pop(next(iter(self._fk_cache)))
        self._fk_cache[cache_key] = np.concatenate([result, positions_arr.flatten()])
        
        if return_joints:
            return result, positions_arr
        return result
    
    def compute_jacobian(self, joint_angles_deg: np.ndarray, 
                        analytic: bool = True) -> np.ndarray:
        """
        计算雅可比矩阵
        
        参数:
            joint_angles_deg: 关节角度（度）
            analytic: True=解析计算（推荐），False=数值微分（备用）
        
        返回:
            雅可比矩阵 shape: (3, n) 或 (6, n)
        """
        cache_key = tuple(np.round(joint_angles_deg, 1))
        
        if analytic:
            # ========== 解析计算（缓存） ==========
            if cache_key in self._jacobian_cache:
                return self._jacobian_cache[cache_key]
            
            joints_rad = joint_angles_deg * np.pi / 180.0
            J = self.analytic_jacobian.compute(joints_rad)
            
            if len(self._jacobian_cache) < self._cache_size:
                self._jacobian_cache[cache_key] = J
            
            return J
        else:
            # ========== 数值微分（备用） ==========
            return self._compute_jacobian_numeric(joint_angles_deg)
    
    def _compute_jacobian_numeric(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        """
        数值微分计算雅可比
        
        使用中心差分: J[:,i] = (f(x+δ) - f(x-δ)) / (2δ)
        精度O(δ²)，但需要2n次正运动学调用
        """
        delta = 0.001
        J = np.zeros((3, self.joint_num), dtype=np.float64)
        
        for i in range(self.joint_num):
            # 正向扰动
            joints_plus = joint_angles_deg.copy()
            joints_plus[i] += delta
            pos_plus = self.forward_kinematics(joints_plus)[:3]
            
            # 负向扰动
            joints_minus = joint_angles_deg.copy()
            joints_minus[i] -= delta
            pos_minus = self.forward_kinematics(joints_minus)[:3]
            
            # 中心差分
            J[:, i] = (pos_plus - pos_minus) / (2 * delta)
        
        return J
    
    def inverse_kinematics(self, target_pose: np.ndarray,
                           initial_joints: np.ndarray = None,
                           tolerance: float = 1e-4,
                           max_iter: int = 50) -> np.ndarray:
        """
        优化的逆运动学（混合求解）
        
        参数:
            target_pose: 目标位姿 [x, y, z] 或 [x, y, z, rx, ry, rz]（米）
            initial_joints: 初始关节角（度），None则使用默认值
            tolerance: 位置误差容限（米），默认0.1mm
            max_iter: 最大迭代次数，默认50
        
        返回:
            关节角度（度）
        
        异常:
            IKSolverError: 求解失败
        """
        target_pos = np.array(target_pose[:3], dtype=np.float64)
        
        if initial_joints is None:
            initial_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 使用混合求解器
        try:
            joints = self.hybrid_ik.solve(
                target_pos, 
                np.array(initial_joints), 
                tolerance, 
                max_iter
            )
            return joints
        except Exception as e:
            raise IKSolverError(f"逆运动学求解失败: {e}")
    
    def plan_trajectory(self, start_deg: np.ndarray, target_deg: np.ndarray,
                       num_points: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numba加速的梯形轨迹规划
        
        参数:
            start_deg: 起始关节角（度）
            target_deg: 目标关节角（度）
            num_points: 期望轨迹点数（可选，用于重采样）
        
        返回:
            (轨迹位置, 轨迹速度), shape: (N, n)
        
        异常:
            TrajectoryError: 输入参数无效
        """
        start_deg = np.array(start_deg)
        target_deg = np.array(target_deg)
        
        if len(start_deg) != len(target_deg):
            raise TrajectoryError("起始和目标关节数不匹配")
        
        # 转换为弧度
        start = start_deg * np.pi / 180.0
        target = target_deg * np.pi / 180.0
        
        # ========== Numba加速规划 ==========
        traj_pos, traj_vel = _plan_all_joints_numba(
            start, target, 
            self.max_vel, self.max_acc, 
            self.dt
        )
        
        # 转换回度
        traj_pos_deg = traj_pos * 180.0 / np.pi
        traj_vel_deg = traj_vel * 180.0 / np.pi
        
        return traj_pos_deg, traj_vel_deg
    
    def check_collision(self, joint_angles_deg: np.ndarray,
                      obstacles: np.ndarray,
                      link_radii: np.ndarray = None,
                      method: str = 'spatial_hash') -> Tuple[bool, List]:
        """
        优化的碰撞检测
        
        参数:
            joint_angles_deg: 当前关节角（度）
            obstacles: 障碍物位置数组 shape: (M, 3)，单位：米
            link_radii: 连杆半径数组 shape: (n,)，默认使用典型值
            method: 'spatial_hash'（稀疏场景）或 'sap'（密集场景）
        
        返回:
            (是否碰撞, 碰撞对列表)
            碰撞对格式: [(连杆索引, 障碍物索引), ...]
        """
        if link_radii is None:
            link_radii = np.array([0.05, 0.04, 0.04, 0.03, 0.03, 0.02])
        
        # 获取连杆位置
        _, positions = self.forward_kinematics(joint_angles_deg, return_joints=True)
        positions = positions.reshape(-1, 3)
        
        # 合并检测对象（连杆 + 障碍物）
        all_positions = np.vstack([positions, obstacles])
        all_radii = np.concatenate([
            link_radii, 
            np.full(len(obstacles), 0.05)  # 障碍物默认半径5cm
        ])
        
        # 选择检测算法
        if method == 'spatial_hash':
            collisions = self.spatial_hash.check_collision_pairs(all_positions, all_radii)
        else:
            # SAP算法
            self.sap_collision.clear()
            for i, (pos, r) in enumerate(zip(all_positions, all_radii)):
                self.sap_collision.update_aabb(i, pos, r)
            collisions = self.sap_collision.check_collisions()
        
        # 过滤：只保留连杆与障碍物的碰撞
        n_links = len(link_radii)
        link_obstacle_collisions = [
            (i, j) for i, j in collisions 
            if (i < n_links) != (j < n_links)  # 一个在连杆侧，一个在障碍物侧
        ]
        
        return len(link_obstacle_collisions) > 0, link_obstacle_collisions
    
    def clear_cache(self):
        """清空所有缓存"""
        self._fk_cache.clear()
        self._jacobian_cache.clear()
        logger.debug("缓存已清空")


# ============================================================
# 性能基准测试
# ============================================================

def benchmark():
    """
    性能对比基准测试
    
    测试内容：
        1. 正运动学（缓存 vs 非缓存）
        2. 雅可比计算（解析 vs 数值微分）
        3. 轨迹规划（Numba加速）
        4. 碰撞检测（空间哈希）
        5. 逆运动学（混合求解）
    """
    import time
    
    print("=" * 70)
    print("  机械臂运动学优化模块 - 性能基准测试")
    print("=" * 70)
    
    # 创建优化求解器
    kin = OptimizedKinematics()
    
    # 测试数据
    joints = np.array([30.0, 45.0, -30.0, 60.0, 0.0, 90.0])
    target = np.array([0.25, 0.1, 0.25])  # 工作空间内的目标
    
    # =========================================================
    # 1. 正运动学测试
    # =========================================================
    print("\n[1] 正运动学 (10000次)")
    
    # 首次计算（无缓存）
    start = time.perf_counter()
    for _ in range(10000):
        kin.clear_cache()  # 清除缓存
        pos = kin.forward_kinematics(joints)
    elapsed_cold = time.perf_counter() - start
    print(f"    无缓存:   {elapsed_cold*1000:.2f}ms ({10000/elapsed_cold:.0f} 次/秒)")
    
    # 缓存命中
    kin.forward_kinematics(joints)  # 预热缓存
    start = time.perf_counter()
    for _ in range(10000):
        pos = kin.forward_kinematics(joints)
    elapsed_hot = time.perf_counter() - start
    print(f"    缓存命中: {elapsed_hot*1000:.2f}ms ({10000/elapsed_hot:.0f} 次/秒)")
    print(f"    缓存加速: {elapsed_cold/elapsed_hot:.1f}x")
    
    # =========================================================
    # 2. 雅可比矩阵计算测试
    # =========================================================
    print("\n[2] 雅可比矩阵计算 (10000次)")
    
    # 数值微分（备用）
    start = time.perf_counter()
    for _ in range(10000):
        J_num = kin._compute_jacobian_numeric(joints)
    elapsed_num = time.perf_counter() - start
    print(f"    数值微分:  {elapsed_num*1000:.2f}ms ({10000/elapsed_num:.0f} 次/秒)")
    
    # 解析雅可比（推荐）
    start = time.perf_counter()
    for _ in range(10000):
        J_ana = kin.compute_jacobian(joints, analytic=True)
    elapsed_ana = time.perf_counter() - start
    print(f"    解析雅可比: {elapsed_ana*1000:.2f}ms ({10000/elapsed_ana:.0f} 次/秒)")
    print(f"    加速比:    {elapsed_num/elapsed_ana:.1f}x")
    
    # =========================================================
    # 3. 轨迹规划测试
    # =========================================================
    print("\n[3] 轨迹规划 (1000次)")
    start = time.perf_counter()
    for _ in range(1000):
        traj_pos, traj_vel = kin.plan_trajectory(joints, joints * 0.5)
    elapsed = time.perf_counter() - start
    print(f"    Numba加速: {elapsed*1000:.2f}ms ({1000/elapsed:.0f} 次/秒)")
    print(f"    轨迹点数:  {len(traj_pos)}")
    
    # =========================================================
    # 4. 碰撞检测测试
    # =========================================================
    print("\n[4] 碰撞检测 (10000次)")
    obstacles = np.random.rand(20, 3) * 0.5  # 20个随机障碍物
    start = time.perf_counter()
    for _ in range(10000):
        collision, pairs = kin.check_collision(joints, obstacles, method='spatial_hash')
    elapsed = time.perf_counter() - start
    print(f"    空间哈希: {elapsed*1000:.2f}ms ({10000/elapsed:.0f} 次/秒)")
    print(f"    障碍物:   {len(obstacles)}个")
    
    # =========================================================
    # 5. 逆运动学测试
    # =========================================================
    print("\n[5] 逆运动学 (1000次)")
    start = time.perf_counter()
    for _ in range(1000):
        ik_solution = kin.inverse_kinematics(target, joints)
    elapsed = time.perf_counter() - start
    print(f"    混合求解: {elapsed*1000:.2f}ms ({1000/elapsed:.0f} 次/秒)")
    
    # 验证解的精度
    result_fk = kin.forward_kinematics(ik_solution)[:3]
    error = np.linalg.norm(result_fk - target)
    print(f"    位置误差: {error*1000:.3f}mm")
    
    print("\n" + "=" * 70)
    print("  测试完成")
    print("=" * 70)


if __name__ == "__main__":
    benchmark()
