"""
机械臂运动学优化模块
优化点：
1. 轨迹规划 - Numba加速 + 并行计算
2. 解析雅可比 - 替代数值微分
3. 逆运动学 - 解析解 + 迭代混合
4. 碰撞检测 - 空间哈希 + SAP算法
"""

import numpy as np
import math
import yaml
import logging
from numba import njit, prange, float64, int32
from functools import lru_cache
from typing import Tuple, Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ====================== Numba加速的轨迹规划 ======================
@njit(cache=True)
def _plan_trapezoid_numba(delta: float, max_vel: float, max_acc: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Numba加速的梯形轨迹规划（单关节）
    
    返回: (位置数组, 速度数组)
    """
    if abs(delta) < 1e-8:
        return np.array([0.0]), np.array([0.0])
    
    direction = 1.0 if delta > 0 else -1.0
    dist = abs(delta)
    accel_dist = (max_vel ** 2) / (2 * max_acc)
    
    # 计算各段时间
    if dist <= 2 * accel_dist:
        peak_vel = math.sqrt(dist * max_acc)
        accel_time = peak_vel / max_acc
        uniform_time = 0.0
    else:
        accel_time = max_vel / max_acc
        uniform_time = (dist - 2 * accel_dist) / max_vel
    
    total_time = 2 * accel_time + uniform_time
    num_steps = max(2, int(total_time / dt) + 1)
    dt_actual = total_time / (num_steps - 1)
    
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    
    for k in range(num_steps):
        t = k * dt_actual
        
        if t <= accel_time:
            # 加速段
            velocities[k] = max_vel * t / accel_time * direction
            positions[k] = 0.5 * max_acc * t * t * direction
        elif t <= accel_time + uniform_time:
            # 匀速段
            t_dec = t - accel_time
            velocities[k] = max_vel * direction
            positions[k] = (accel_dist + max_vel * t_dec) * direction
        else:
            # 减速段
            t_dec = t - accel_time - uniform_time
            remaining = dist - accel_dist
            velocities[k] = (max_vel - max_acc * t_dec) * direction
            positions[k] = (remaining + max_vel * t_dec - 0.5 * max_acc * t_dec * t_dec) * direction
    
    positions[-1] = delta
    velocities[-1] = 0.0
    
    return positions, velocities


@njit(cache=True, parallel=True)
def _plan_all_joints_numba(start: np.ndarray, target: np.ndarray, 
                          max_vel: np.ndarray, max_acc: np.ndarray, 
                          dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Numba并行规划所有关节
    
    返回: (轨迹位置, 轨迹速度), shape: (N, 6)
    """
    n_joints = len(start)
    max_points = 2000  # 最大轨迹点数
    
    # 预分配结果数组
    traj_pos = np.zeros((max_points, n_joints))
    traj_vel = np.zeros((max_points, n_joints))
    
    # 分别规划每个关节
    for i in prange(n_joints):
        pos, vel = _plan_trapezoid_numba(target[i] - start[i], max_vel[i], max_acc[i], dt)
        n_pts = len(pos)
        for j in range(n_pts):
            traj_pos[j, i] = start[i] + pos[j]
            traj_vel[j, i] = vel[j]
    
    # 找到实际长度并截取
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


# ====================== 解析雅可比计算 ======================
class AnalyticJacobian:
    """六轴机械臂解析雅可比计算器
    
    基于标准D-H参数的解析雅可比，避免数值微分的精度损失和计算开销
    """
    
    def __init__(self, dh_fixed: dict):
        self.dh_fixed = dh_fixed
        self.n = len(dh_fixed)
        self._precompute_chain_params()
    
    def _precompute_chain_params(self):
        """预计算链式参数（沿用PyBullet风格）"""
        self.link_lengths = np.array([p['a'] for p in self.dh_fixed.values()])
        self.joint_offsets = np.array([p['d'] for p in self.dh_fixed.values()])
        self.twist_angles = np.array([p['alpha_rad'] for p in self.dh_fixed.values()])
    
    def compute(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """计算解析雅可比矩阵
        
        Args:
            joint_angles_rad: 关节角度（弧度），shape: (6,)
        
        Returns:
            J: 位置雅可比矩阵，shape: (3, 6)
        """
        J = np.zeros((3, self.n), dtype=np.float64)
        
        # 计算各关节位置和姿态
        positions = []
        z_axes = []
        pos_total = np.zeros(3)
        
        T = np.eye(4, dtype=np.float64)
        
        for i in range(self.n):
            a = self.link_lengths[i]
            alpha = self.twist_angles[i]
            d = self.joint_offsets[i]
            theta = joint_angles_rad[i]
            
            # D-H变换
            ca, sa = math.cos(alpha), math.sin(alpha)
            ct, st = math.cos(theta), math.sin(theta)
            
            Ti = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            
            T = T @ Ti
            positions.append(T[:3, 3].copy())
            z_axes.append(T[:3, 2].copy())
        
        # 末端位置
        ee_pos = positions[-1]
        
        # 计算雅可比列
        for i in range(self.n):
            # 位置雅可比: Jp_i = z_{i-1} × (p_ee - p_{i-1})
            z_prev = z_axes[i-1] if i > 0 else np.array([0., 0., 1.])
            p_diff = ee_pos - positions[i-1] if i > 0 else ee_pos
            J[:, i] = np.cross(z_prev, p_diff)
        
        return J
    
    def compute_full(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """计算完整6x6雅可比（包含位置和姿态）"""
        J_pos = self.compute(joint_angles_rad)
        J_ori = self._compute_orientation_jacobian(joint_angles_rad)
        return np.vstack([J_pos, J_ori])
    
    def _compute_orientation_jacobian(self, joint_angles_rad: np.ndarray) -> np.ndarray:
        """计算姿态雅可比"""
        J_ori = np.zeros((3, self.n), dtype=np.float64)
        
        z_prev = np.array([0., 0., 1.])
        for i in range(self.n):
            J_ori[:, i] = z_prev
            if i < self.n - 1:
                # 累积旋转
                alpha = self.twist_angles[i]
                ca, sa = math.cos(alpha), math.sin(alpha)
                theta = joint_angles_rad[i]
                ct, st = math.cos(theta), math.sin(theta)
                # 简化：直接使用下一个关节的z轴
                z_prev = np.array([-st*sa, ct*sa, ca])
        
        return J_ori


# ====================== 混合逆运动学求解器 ======================
class HybridIKSolver:
    """混合逆运动学求解器
    
    结合解析闭式解和迭代优化的优势：
    1. 前3个关节使用解析解（位置求解）
    2. 后3个关节使用迭代优化
    """
    
    def __init__(self, dh_fixed: dict):
        self.dh_fixed = dh_fixed
        self.n = len(dh_fixed)
        self._extract_geometric_params()
    
    def _extract_geometric_params(self):
        """提取几何参数"""
        self.l1 = self.dh_fixed[1]['a'] / 1000 + self.dh_fixed[2]['a'] / 1000  # 大臂
        self.l2 = self.dh_fixed[3]['a'] / 1000 + self.dh_fixed[4]['a'] / 1000  # 小臂
        self.l3 = self.dh_fixed[5]['d'] / 1000  # 腕部偏移
        self.h = self.dh_fixed[0]['d'] / 1000 + self.dh_fixed[4]['d'] / 1000  # 高度偏移
    
    def solve(self, target_pos: np.ndarray, initial_joints: np.ndarray, 
              tolerance: float = 1e-4, max_iter: int = 50) -> np.ndarray:
        """
        混合逆运动学求解
        
        Args:
            target_pos: 目标位置 [x, y, z] (米)
            initial_joints: 初始关节角（度）
            tolerance: 位置误差容限（米）
            max_iter: 最大迭代次数
        
        Returns:
            关节角数组（度）
        """
        # 解析求解前3个关节
        j1, j2, j3 = self._solve_spherical_wrist(target_pos)
        
        # 初始关节角（度转弧度）
        current = np.array(initial_joints, dtype=np.float64) * np.pi / 180.0
        current[0] = j1 if j1 is not None else current[0]
        current[1] = j2 if j2 is not None else current[1]
        current[2] = j3 if j3 is not None else current[2]
        
        # 迭代优化所有关节
        current = self._iterative_refinement(current, target_pos, tolerance, max_iter)
        
        return current * 180.0 / np.pi
    
    def _solve_spherical_wrist(self, target: np.ndarray) -> Tuple[float, float, float]:
        """
        解析求解球形手腕机械臂的前3个关节
        
        基于几何关系求解肩部和肘部关节角
        """
        x, y, z = target[0], target[1], target[2]
        
        # 关节1: 基座旋转 (绕z轴)
        j1 = math.atan2(y, x) if abs(x) > 1e-6 or abs(y) > 1e-6 else 0.0
        
        # 水平距离和高度
        r = math.sqrt(x**2 + y**2)
        h = z - self.h
        
        # 求解2-3关节（平面几何）
        D = (r**2 + h**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        
        if abs(D) > 1.0:
            # 目标超出工作范围
            return j1, None, None
        
        # 肘部角度（两种解：肘下/肘上）
        elbow_angle = math.atan2(-math.sqrt(1 - D**2), D)
        
        # 肩部角度
        alpha = math.atan2(h, r)
        beta = math.atan2(self.l2 * math.sin(elbow_angle), 
                         self.l1 + self.l2 * math.cos(elbow_angle))
        j2 = alpha - beta
        
        # 关节3（调整腕部方向）
        j3 = -j2 - elbow_angle
        
        return j1, j2, j3
    
    def _iterative_refinement(self, joints: np.ndarray, target: np.ndarray,
                              tol: float, max_iter: int) -> np.ndarray:
        """使用阻尼最小二乘法迭代优化"""
        from numpy.linalg import pinv
        
        # 预计算雅可比
        jacobian = AnalyticJacobian(self.dh_fixed)
        
        for _ in range(max_iter):
            # 计算当前位置
            T = self._compute_fk(joints)
            current_pos = T[:3, 3]
            
            # 误差
            error = target - current_pos
            if np.linalg.norm(error) < tol:
                break
            
            # 计算雅可比
            J = jacobian.compute(joints)
            
            # 阻尼最小二乘
            lamda = 0.01
            JtJ = J @ J.T + lamda * np.eye(3)
            delta = J.T @ np.linalg.solve(JtJ, error)
            
            # 更新（带限幅）
            joints = joints + delta * 0.5
            joints = np.clip(joints, -np.pi, np.pi)
        
        return joints
    
    def _compute_fk(self, joints: np.ndarray) -> np.ndarray:
        """计算正运动学"""
        T = np.eye(4, dtype=np.float64)
        
        for i in range(len(self.dh_fixed)):
            params = self.dh_fixed[i]
            a, alpha = params['a'], params['alpha_rad']
            d, theta_base = params['d'], params['theta_base']
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


# ====================== 空间哈希碰撞检测 ======================
class SpatialHashCollision:
    """空间哈希碰撞检测器
    
    使用均匀网格空间哈希实现O(n)平均复杂度的碰撞检测，
    替代O(n²)的穷举检测
    """
    
    def __init__(self, cell_size: float = 0.05):
        """
        Args:
            cell_size: 网格单元大小（米），通常设为最大碰撞体尺寸的2倍
        """
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self._grid = {}  # 空间哈希表
    
    def _hash_pos(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """计算位置的哈希键"""
        ix = int(math.floor(pos[0] * self.inv_cell_size))
        iy = int(math.floor(pos[1] * self.inv_cell_size))
        iz = int(math.floor(pos[2] * self.inv_cell_size))
        return (ix, iy, iz)
    
    def clear(self):
        """清空哈希表"""
        self._grid.clear()
    
    def insert(self, obj_id: int, pos: np.ndarray, radius: float):
        """插入碰撞体到空间哈希"""
        # 计算物体占据的网格单元
        r = radius * self.inv_cell_size
        for dx in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
            for dy in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
                for dz in range(int(math.floor(-r)), int(math.ceil(r)) + 1):
                    ix = int(math.floor(pos[0] * self.inv_cell_size)) + dx
                    iy = int(math.floor(pos[1] * self.inv_cell_size)) + dy
                    iz = int(math.floor(pos[2] * self.inv_cell_size)) + dz
                    key = (ix, iy, iz)
                    if key not in self._grid:
                        self._grid[key] = []
                    self._grid[key].append((obj_id, pos, radius))
    
    def query_nearby(self, pos: np.ndarray, radius: float) -> List[Tuple]:
        """查询附近可能的碰撞对"""
        nearby = []
        r = radius * self.inv_cell_size
        center = self._hash_pos(pos)
        
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
        
        Args:
            positions: 物体位置数组 shape: (n, 3)
            radii: 物体半径数组 shape: (n,)
        
        Returns:
            碰撞对列表 [(i, j), ...]
        """
        self.clear()
        n = len(positions)
        
        # 构建空间哈希
        for i in range(n):
            self.insert(i, positions[i], radii[i])
        
        # 检测碰撞
        checked = set()
        collisions = []
        
        for i in range(n):
            nearby = self.query_nearby(positions[i], radii[i])
            
            for j, pos_j, r_j in nearby:
                if i >= j:
                    continue
                
                pair_key = (min(i, j), max(i, j))
                if pair_key in checked:
                    continue
                checked.add(pair_key)
                
                # 精确碰撞检测
                dist = np.linalg.norm(positions[i] - pos_j)
                min_dist = radii[i] + r_j
                
                if dist < min_dist:
                    collisions.append((i, j))
        
        return collisions


# ====================== SAP碰撞检测（扫描轴） ======================
class SAPCollision:
    """
    SAP (Sweep and Prune) 碰撞检测
    
    利用AABB和轴对齐包围盒实现高效的一维排序碰撞检测
    适用于刚体数量较多的场景
    """
    
    def __init__(self):
        self._aabbs = []  # AABB包围盒列表
        self._active = []  # 活跃物体列表
    
    def update_aabb(self, obj_id: int, pos: np.ndarray, radius: float):
        """更新物体AABB"""
        # 简单球体AABB
        self._aabbs.append({
            'id': obj_id,
            'min': pos - radius,
            'max': pos + radius,
            'radius': radius,
            'pos': pos
        })
    
    def clear(self):
        self._aabbs.clear()
        self._active.clear()
    
    def check_collisions(self) -> List[Tuple[int, int]]:
        """
        SAP碰撞检测
        
        Returns:
            碰撞对列表
        """
        if len(self._aabbs) < 2:
            return []
        
        # 按x轴坐标排序
        sorted_aabbs = sorted(self._aabbs, key=lambda a: (a['min'][0], a['id']))
        
        collisions = []
        active = []  # 当前活跃的AABB
        
        for aabb in sorted_aabbs:
            aabb_min = aabb['min']
            aabb_max = aabb['max']
            
            # 移除不在范围内的活跃物体
            new_active = []
            for active_aabb in active:
                if active_aabb['max'][0] < aabb_min[0]:
                    continue
                new_active.append(active_aabb)
            active = new_active
            
            # 与活跃物体检测碰撞
            for active_aabb in active:
                if self._aabb_overlap(active_aabb, aabb):
                    # 精确碰撞检测
                    dist = np.linalg.norm(active_aabb['pos'] - aabb['pos'])
                    min_dist = active_aabb['radius'] + aabb['radius']
                    if dist < min_dist:
                        collisions.append((active_aabb['id'], aabb['id']))
            
            active.append(aabb)
        
        return collisions
    
    def _aabb_overlap(self, a: dict, b: dict) -> bool:
        """检测两个AABB是否重叠"""
        return (a['min'][0] <= b['max'][0] and a['max'][0] >= b['min'][0] and
                a['min'][1] <= b['max'][1] and a['max'][1] >= b['min'][1] and
                a['min'][2] <= b['max'][2] and a['max'][2] >= b['min'][2])


# ====================== 优化的运动学主类 ======================
class OptimizedKinematics:
    """优化版机械臂运动学解算器"""
    
    def __init__(self, config_path: str = "config/arm_config.yaml"):
        self.config_path = config_path
        self.dh_params = {}
        self.joint_limits = {}
        self.joint_num = 6
        
        # 预计算缓存
        self._fk_cache = {}
        self._jacobian_cache = {}
        self._cache_size = 1000
        
        # 加载配置
        self._load_config()
        self._precompute()
        
        # 初始化优化组件
        self.analytic_jacobian = AnalyticJacobian(self.dh_fixed)
        self.hybrid_ik = HybridIKSolver(self.dh_fixed)
        self.spatial_hash = SpatialHashCollision(cell_size=0.1)
        self.sap_collision = SAPCollision()
        
        # 轨迹规划参数
        self.max_vel = np.array([60, 45, 45, 90, 90, 120]) * np.pi / 180.0  # rad/s
        self.max_acc = np.array([120, 90, 90, 180, 180, 240]) * np.pi / 180.0  # rad/s^2
        self.dt = 0.001  # 1ms控制周期
    
    def _load_config(self):
        """加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.dh_params = config['DH_PARAMS']
            self.joint_limits = config['JOINT_LIMITS']
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    def _precompute(self):
        """预计算D-H固定参数"""
        self.dh_fixed = {}
        for joint, params in self.dh_params.items():
            idx = int(joint.replace('joint', '')) - 1
            self.dh_fixed[idx] = {
                'a': params['a'] / 1000,
                'alpha_rad': math.radians(params['alpha']),
                'd': params['d'] / 1000,
                'theta_base': params['theta']
            }
    
    def forward_kinematics(self, joint_angles_deg: np.ndarray, 
                          return_joints: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        优化的正运动学（向量化 + 缓存）
        
        Args:
            joint_angles_deg: 关节角度（度）
            return_joints: 是否返回中间关节位置
        
        Returns:
            末端位姿 [x, y, z, rx, ry, rz] 或 (末端位姿, 中间关节位置)
        """
        # 缓存键
        cache_key = tuple(np.round(joint_angles_deg, 1))
        if cache_key in self._fk_cache:
            cached = self._fk_cache[cache_key]
            if return_joints:
                return cached[:6], cached[6:]  # 返回位姿和关节位置
            return cached[:6]
        
        # 弧度转换
        joints_rad = joint_angles_deg * np.pi / 180.0
        
        # 批量计算变换矩阵
        T = np.eye(4, dtype=np.float64)
        positions = []
        
        for i in range(self.joint_num):
            params = self.dh_fixed[i]
            a, alpha = params['a'], params['alpha_rad']
            d, theta = params['d'], params['theta_base'] * np.pi / 180.0 + joints_rad[i]
            
            ca, sa = math.cos(alpha), math.sin(alpha)
            ct, st = math.cos(theta), math.sin(theta)
            
            T = T @ np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            positions.append(T[:3, 3].copy())
        
        # 提取位姿
        pos = T[:3, 3]
        r31, r32, r33 = T[2, 0], T[2, 1], T[2, 2]
        r21, r11 = T[1, 0], T[0, 0]
        
        euler = np.array([
            math.degrees(math.atan2(r32, r33)),
            math.degrees(math.atan2(-r31, math.hypot(r32, r33))),
            math.degrees(math.atan2(r21, r11))
        ])
        
        result = np.concatenate([pos, euler])
        positions_arr = np.array(positions)
        
        # 缓存管理（存储位姿 + 关节位置）
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
        
        Args:
            joint_angles_deg: 关节角度（度）
            analytic: True=解析计算, False=数值微分
        
        Returns:
            雅可比矩阵 shape: (3, 6) 或 (6, 6)
        """
        cache_key = tuple(np.round(joint_angles_deg, 1))
        
        if analytic:
            # 解析计算
            if cache_key in self._jacobian_cache:
                return self._jacobian_cache[cache_key]
            
            joints_rad = joint_angles_deg * np.pi / 180.0
            J = self.analytic_jacobian.compute(joints_rad)
            
            if len(self._jacobian_cache) < self._cache_size:
                self._jacobian_cache[cache_key] = J
            
            return J
        else:
            # 数值微分（备用）
            return self._compute_jacobian_numeric(joint_angles_deg)
    
    def _compute_jacobian_numeric(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        """数值微分计算雅可比"""
        delta = 0.001
        J = np.zeros((3, self.joint_num), dtype=np.float64)
        pos0 = self.forward_kinematics(joint_angles_deg)[:3]
        
        for i in range(self.joint_num):
            joints_plus = joint_angles_deg.copy()
            joints_plus[i] += delta
            pos_plus = self.forward_kinematics(joints_plus)[:3]
            
            joints_minus = joint_angles_deg.copy()
            joints_minus[i] -= delta
            pos_minus = self.forward_kinematics(joints_minus)[:3]
            
            J[:, i] = (pos_plus - pos_minus) / (2 * delta)
        
        return J
    
    def inverse_kinematics(self, target_pose: np.ndarray,
                           initial_joints: np.ndarray = None,
                           tolerance: float = 1e-4,
                           max_iter: int = 50) -> np.ndarray:
        """
        优化的逆运动学（混合求解）
        
        Args:
            target_pose: 目标位姿 [x, y, z] 或 [x, y, z, rx, ry, rz]
            initial_joints: 初始关节角（度）
            tolerance: 位置误差容限（米）
            max_iter: 最大迭代次数
        
        Returns:
            关节角度（度）
        """
        target_pos = np.array(target_pose[:3], dtype=np.float64)
        
        if initial_joints is None:
            initial_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 使用混合求解器
        joints = self.hybrid_ik.solve(target_pos, np.array(initial_joints), 
                                       tolerance, max_iter)
        
        return joints
    
    def plan_trajectory(self, start_deg: np.ndarray, target_deg: np.ndarray,
                       num_points: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        优化的轨迹规划（Numba加速）
        
        Args:
            start_deg: 起始关节角（度）
            target_deg: 目标关节角（度）
            num_points: 期望轨迹点数（可选）
        
        Returns:
            (轨迹位置, 轨迹速度), shape: (N, 6)
        """
        start = start_deg * np.pi / 180.0
        target = target_deg * np.pi / 180.0
        
        # 使用Numba加速的规划
        traj_pos, traj_vel = _plan_all_joints_numba(start, target, 
                                                     self.max_vel, self.max_acc, 
                                                     self.dt)
        
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
        
        Args:
            joint_angles_deg: 当前关节角（度）
            obstacles: 障碍物位置数组 shape: (M, 3)
            link_radii: 连杆半径数组 shape: (n,)
            method: 'spatial_hash' 或 'sap'
        
        Returns:
            (是否碰撞, 碰撞对列表)
        """
        if link_radii is None:
            link_radii = np.array([0.05, 0.04, 0.04, 0.03, 0.03, 0.02])
        
        # 获取连杆位置
        _, positions = self.forward_kinematics(joint_angles_deg, return_joints=True)
        positions = positions.reshape(-1, 3)  # 确保是 (n, 3) 形状
        
        # 合并末端和障碍物
        all_positions = np.vstack([positions, obstacles])
        all_radii = np.concatenate([link_radii, np.full(len(obstacles), 0.05)])
        
        if method == 'spatial_hash':
            collisions = self.spatial_hash.check_collision_pairs(all_positions, all_radii)
        else:
            # SAP检测
            self.sap_collision.clear()
            for i, (pos, r) in enumerate(zip(all_positions, all_radii)):
                self.sap_collision.update_aabb(i, pos, r)
            collisions = self.sap_collision.check_collisions()
        
        # 过滤连杆与障碍物的碰撞
        n_links = len(link_radii)
        link_obstacle_collisions = [(i, j) for i, j in collisions 
                                    if (i < n_links) != (j < n_links)]
        
        return len(link_obstacle_collisions) > 0, link_obstacle_collisions
    
    def clear_cache(self):
        """清空缓存"""
        self._fk_cache.clear()
        self._jacobian_cache.clear()


# ====================== 性能基准测试 ======================
def benchmark():
    """性能对比基准测试"""
    import time
    
    print("=" * 60)
    print("运动学优化性能基准测试")
    print("=" * 60)
    
    # 创建优化求解器
    kin = OptimizedKinematics()
    
    # 测试数据
    joints = np.array([30.0, 45.0, -30.0, 60.0, 0.0, 90.0])
    target = np.array([0.3, 0.1, 0.4])
    
    # 1. 正运动学测试
    print("\n1. 正运动学 (10000次)")
    start = time.perf_counter()
    for _ in range(10000):
        pos = kin.forward_kinematics(joints)
    elapsed = time.perf_counter() - start
    print(f"   优化版本: {elapsed*1000:.2f}ms ({10000/elapsed:.0f} 次/秒)")
    
    # 2. 雅可比计算测试
    print("\n2. 雅可比矩阵计算 (10000次)")
    
    # 数值微分
    start = time.perf_counter()
    for _ in range(10000):
        J_num = kin._compute_jacobian_numeric(joints)
    elapsed_num = time.perf_counter() - start
    print(f"   数值微分: {elapsed_num*1000:.2f}ms ({10000/elapsed_num:.0f} 次/秒)")
    
    # 解析雅可比
    start = time.perf_counter()
    for _ in range(10000):
        J_ana = kin.compute_jacobian(joints, analytic=True)
    elapsed_ana = time.perf_counter() - start
    print(f"   解析雅可比: {elapsed_ana*1000:.2f}ms ({10000/elapsed_ana:.0f} 次/秒)")
    print(f"   加速比: {elapsed_num/elapsed_ana:.1f}x")
    
    # 3. 轨迹规划测试
    print("\n3. 轨迹规划 (1000次)")
    start = time.perf_counter()
    for _ in range(1000):
        traj_pos, traj_vel = kin.plan_trajectory(joints, joints * 0.5)
    elapsed = time.perf_counter() - start
    print(f"   优化版本: {elapsed*1000:.2f}ms ({1000/elapsed:.0f} 次/秒)")
    print(f"   轨迹点数: {len(traj_pos)}")
    
    # 4. 碰撞检测测试
    print("\n4. 碰撞检测 (10000次)")
    obstacles = np.random.rand(20, 3) * 0.5
    start = time.perf_counter()
    for _ in range(10000):
        collision, pairs = kin.check_collision(joints, obstacles, method='spatial_hash')
    elapsed = time.perf_counter() - start
    print(f"   空间哈希: {elapsed*1000:.2f}ms ({10000/elapsed:.0f} 次/秒)")
    
    # 5. 逆运动学测试
    print("\n5. 逆运动学 (1000次)")
    start = time.perf_counter()
    for _ in range(1000):
        ik_solution = kin.inverse_kinematics(target, joints)
    elapsed = time.perf_counter() - start
    print(f"   混合求解: {elapsed*1000:.2f}ms ({1000/elapsed:.0f} 次/秒)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark()
