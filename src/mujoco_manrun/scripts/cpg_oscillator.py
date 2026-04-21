import numpy as np

class CPGOscillator:
    """CPG中枢模式发生器（独立算法，无业务依赖）"""
    def __init__(self, freq=0.5, amp=0.4, phase=0.0, coupling_strength=0.2):
        self.base_freq = freq        # 基础频率
        self.base_amp = amp          # 基础振幅
        self.base_phase = phase      # 初始相位
        self.base_coupling = coupling_strength  # 基础耦合强度
        
        # 动态参数
        self.freq = freq
        self.amp = amp
        self.phase = phase
        self.coupling = coupling_strength
        self.state = np.array([np.sin(phase), np.cos(phase)])  # 振荡器状态(x,y)

    def update(self, dt, target_phase=0.0, speed_factor=1.0, turn_factor=0.0):
        """
        更新CPG状态
        :param dt: 时间步长
        :param target_phase: 目标相位（用于耦合）
        :param speed_factor: 速度系数（0~1）
        :param turn_factor: 转向系数（-1~1）
        :return: 关节目标偏移量
        """
        # 动态调整耦合强度：速度/转向越大，耦合越强
        self.coupling = self.base_coupling * (1.0 + 0.5 * speed_factor + 0.8 * abs(turn_factor))
        self.coupling = np.clip(self.coupling, 0.1, 0.5)

        # 范德波尔振荡器方程（生物节律更自然）
        mu = 1.0  # 非线性系数
        x, y = self.state
        dx = 2 * np.pi * self.freq * y + self.coupling * np.sin(target_phase - self.phase)
        dy = 2 * np.pi * self.freq * (mu * (1 - x**2) * y - x)
        
        # 积分更新状态
        self.state += np.array([dx, dy]) * dt
        self.phase = np.arctan2(self.state[0], self.state[1])  # 更新相位
        
        # 返回当前输出
        return self.amp * self.state[0]

    def reset(self):
        """重置CPG到初始状态"""
        self.freq = self.base_freq
        self.amp = self.base_amp
        self.coupling = self.base_coupling
        self.phase = self.base_phase
        self.state = np.array([np.sin(self.phase), np.cos(self.phase)])
