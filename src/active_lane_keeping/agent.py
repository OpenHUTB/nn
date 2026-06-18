from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Dict

from data_logger import DataLogger


class Agent():
    """Agent that outputs the desired behaviour given observations.
    
    Supports multiple controllers:
    - simple: Hard coded controller
    - p: Proportional controller
    - pd: Proportional-Derivative controller
    - pid: PID controller
    - pid_gs: PID with Gain Scheduling (velocity-based parameter adjustment)
    - mpc: Model Predictive Control
    """

    # 增益调度参数表：速度区间 -> (tau_p, tau_d, tau_i)
    GAIN_SCHEDULE: Dict[str, tuple] = {
        'low': (0.8, 0.02, 0.00000003),    # < 30 km/h
        'medium': (0.65, 0.034, 0.00000002), # 30-60 km/h
        'high': (0.5, 0.05, 0.00000001)      # > 60 km/h
    }
    
    # 速度阈值（m/s转换为km/h需要乘3.6）
    SPEED_THRESHOLD_LOW = 30 / 3.6  # ~8.33 m/s
    SPEED_THRESHOLD_HIGH = 60 / 3.6  # ~16.67 m/s
    
    # MPC参数
    MPC_PREDICTION_HORIZON = 5  # 预测步数
    MPC_CONTROL_HORIZON = 3    # 控制时域

    def __init__(self, tau_p:float = 0.65, tau_d:float = 0.034, tau_i:float = 0.00000002,
        surface_lower_threshold:float = 20000000.0, throttle:float = 0.3,
        surface_upper_threshold=30000000.0, controller:str = 'pid',
        use_gain_scheduling:bool = False, use_adaptive_integral:bool = True,
        mpc_horizon:int = 5, mpc_control_horizon:int = 3) -> None:
        """Constructor

        Args:
            tau_p (float, optional): Parameter for the proportional part of the
                PID-Controller. Defaults to 0.65.
            tau_d (float, optional): Parameter for the derivative part of the
                PID-Controller. Defaults to 0.034.
            tau_i (float, optional): Parameter for the integral part of the
                PID-Controller. Defaults to 0.00000002.
            surface_lower_threshold (float, optional): Minimum amount of surface
                that has to be detected in order to calculate new output. This
                helps to find suitable lanes. Defaults to 20000000.0.
            throttle (float, optional): Throttle to return at each time step.
                Defaults to 0.3.
            surface_upper_threshold (float, optional): Maximum amount of surface
                that has to be detected in order to calculate new output. This
                helps to find suitable lanes. Defaults to 30000000.0.
            controller (str, optional): Identifies the controller to be used.
                This can be one of the following:
                    - 'simple': hard coded controller that does not use any of
                        the tau parameters
                    - 'p': controller that only uses the proportional part
                    - 'pd': controller that only uses the proportional and
                        derivative part
                    - 'pid': pid-controller
                    - 'pid_gs': pid-controller with gain scheduling
                    - 'mpc': model predictive control
                Defaults to 'pid'.
            use_gain_scheduling (bool, optional): Enable gain scheduling for PID.
                Defaults to False.
            use_adaptive_integral (bool, optional): Enable adaptive integral
                saturation. Defaults to True.
            mpc_horizon (int, optional): Prediction horizon for MPC.
                Defaults to 5.
            mpc_control_horizon (int, optional): Control horizon for MPC.
                Defaults to 3.
        """
        self.tau_p = tau_p
        self.tau_d = tau_d
        self.tau_i = tau_i
        self.surface_lower_threshold = surface_lower_threshold
        self.surface_upper_threshold = surface_upper_threshold
        self.prev_error = None
        self.throttle = throttle
        self.func = None
        self.errors = []
        self.controller_name = controller
        self.use_gain_scheduling = use_gain_scheduling
        self.use_adaptive_integral = use_adaptive_integral
        self.mpc_horizon = mpc_horizon
        self.mpc_control_horizon = mpc_control_horizon
        
        # 自适应积分饱和参数
        self.integral_limit = 50.0  # 积分累积上限
        self.integral = 0.0         # 当前积分值
        
        # MPC相关
        self.error_history = []      # 误差历史用于MPC预测
        self.mpc_history_len = 10    # MPC预测所需的历史长度
        
        # 速度（用于增益调度）
        self.current_speed = 0.0
        
        self._select_controller_method(name=self.controller_name)
        self.data_logger: Optional[DataLogger] = None
        self.current_step = 0

    @classmethod
    def from_config(cls, config: dict) -> 'Agent':
        """Create an Agent instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing controller settings.

        Returns:
            Agent: A new Agent instance configured according to the provided config.
        """
        ctrl = config.get('controller', {})
        
        return cls(
            tau_p=ctrl.get('tau_p', 0.65),
            tau_d=ctrl.get('tau_d', 0.034),
            tau_i=ctrl.get('tau_i', 0.00000002),
            throttle=ctrl.get('throttle', 0.3),
            surface_lower_threshold=ctrl.get('surface_lower_threshold', 20000000.0),
            surface_upper_threshold=ctrl.get('surface_upper_threshold', 30000000.0),
            controller=ctrl.get('default_controller', 'pid'),
            use_gain_scheduling=ctrl.get('use_gain_scheduling', False),
            use_adaptive_integral=ctrl.get('use_adaptive_integral', True),
            mpc_horizon=ctrl.get('mpc_horizon', 5),
            mpc_control_horizon=ctrl.get('mpc_control_horizon', 3)
        )

    def set_data_logger(self, logger: DataLogger) -> None:
        """Set the data logger for recording step data.

        Args:
            logger (DataLogger): DataLogger instance to use for recording.
        """
        self.data_logger = logger
        self.data_logger.set_metadata(
            controller=self.controller_name,
            tau_p=self.tau_p,
            tau_d=self.tau_d,
            tau_i=self.tau_i,
            throttle=self.throttle
        )

    def _select_controller_method(self, name:str) -> None:
        """Helper method to select the controller

        Args:
            name (str): Name of the controller.

        Raises:
            Exception: If the given name does not map to a controller method.
        """
        name = name.lower()
        if name == 'simple':
            self.func = Agent._simple_controller
        elif name == 'p':
            self.func = self._p_controller
        elif name == 'pd':
            self.func = self._pd_controller
        elif name == 'pid':
            self.func = self._pid_controller
        elif name == 'pid_gs':
            self.func = self._pid_gs_controller
        elif name == 'mpc':
            self.func = self._mpc_controller
        else:
            raise Exception(f'Controller name \'{name}\' is not applicable.')

    def check_surface_area(self, detection_surface_area:float) -> bool:
        """Checks Surface Area

        Args:
            detection_surface_area (float): _description_

        Returns:
            bool: Whether the given surface area is not inbetween the selected
                interval.
        """
        lower = (detection_surface_area < self.surface_lower_threshold)
        upper = (detection_surface_area > self.surface_upper_threshold)
        return lower or upper

    def show_error(self) -> None:
        """Displays Error

        Displays the difference to the center of all past steps.
        """
        plt.figure(1)
        plt.clf()
        x = range(len(self.errors))
        y = self.errors
        plt.plot(x, y, 'g', label=self.controller_name)
        plt.plot(x, np.zeros(len(self.errors)), 'r', label='baseline')
        plt.xlabel('Step')
        plt.ylabel('Difference to Center')
        plt.legend()
        plt.pause(1e-10)

    def save_error_fig(self, path:str, run_id:str) -> None:
        """Save the Figure of the Error Plot

        Args:
            path (str): Folder to place the file in.
            run_id (str): Unique identifier to prevent overwriting files.
        """
        plt.figure(1)
        file_name = os.path.join(path, f'{run_id}_error.jpg')
        plt.savefig(file_name)

    def get_actions(self, detection_surface_area:float,
        error:float, speed:float = 0.0) -> tuple[float, float]:
        """Retrieve action based on Observation

        Args:
            error (float): Difference to the center of the detected lane.
            detection_surface_area (float): Detected surface area.
            speed (float): Current vehicle speed in m/s.

        Returns:
            tuple[float, float]:
                [0]: Steering angle to use.
                [1]: Throttle to apply.
        """
        # 更新速度（用于增益调度）
        self.current_speed = abs(speed)
        
        # 记录误差历史（用于MPC）
        self.error_history.append(error)
        if len(self.error_history) > self.mpc_history_len:
            self.error_history.pop(0)
        
        if (self.check_surface_area(detection_surface_area)):
            steer = 0
            if len(self.errors) > 0:
                self.errors.append(self.errors[-1])
            else:
                self.errors.append(0)
        else:
            self.errors.append(error)
            
            # 根据控制器类型调用不同的控制方法
            if self.controller_name == 'pid_gs':
                steer = self.func(error=error, speed=self.current_speed)
            elif self.controller_name == 'mpc':
                steer = self.func(error=error, speed=self.current_speed)
            else:
                steer = self.func(error=error)

        if self.data_logger:
            self.data_logger.record_step(
                step=self.current_step,
                error=error,
                steer=steer,
                throttle=self.throttle,
                detection_surface_area=detection_surface_area
            )
            self.current_step += 1

        return steer, self.throttle

    def save_data(self) -> None:
        """Save all recorded data using the data logger."""
        if self.data_logger:
            self.data_logger.save_all()
            self.data_logger.save_summary()

    @staticmethod
    def _simple_controller(error:float) -> float:
        """Hard Coded Controller

        Args:
            error (float): Difference to the center of the detected lane.

        Returns:
            float: Steering angle to use.
        """
        if (abs(error) < 0.1):
            steer = 0
        elif error > 0:
            steer = -0.75
        else:
            steer = 0.75
        return steer

    def _p_controller(self, error:float) -> float:
        """Proportional Controller

        Args:
            error (float): Difference to the center of the detected lane.

        Returns:
            float: Steering angle to use.
        """
        steer = - self.tau_p * error
        return steer

    def _pd_controller(self, error:float) -> float:
        """Proportional and Derivative Controller

        Args:
            error (float): Difference to the center of the detected lane.

        Returns:
            float: Steering angle to use.
        """
        if len(self.errors) > 2:
            self.prev_error = self.errors[-2]
        else:
            self.prev_error = self.errors[-1]
        
        deviation = error - self.prev_error
        steer = - self.tau_d * deviation/0.1 + self._p_controller(error)
        return steer

    def _pid_controller(self, error:float) -> float:
        """PID-Controller with optional adaptive integral saturation

        Args:
            error (float): Difference to the center of the detected lane.

        Returns:
            float: Steering angle to use.
        """
        # 自适应积分饱和
        if self.use_adaptive_integral:
            self.integral += error
            # 限制积分累积，防止积分饱和
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
            # 根据速度调整积分限幅（速度越高，积分限幅越小）
            adaptive_limit = self.integral_limit / (1 + self.current_speed * 0.1)
            self.integral = np.clip(self.integral, -adaptive_limit, adaptive_limit)
            sum_error = self.integral
        else:
            sum_error = sum(self.errors)
        
        steer = - self.tau_i * sum_error + self._pd_controller(error)
        return steer
    
    def _pid_gs_controller(self, error:float, speed:float) -> float:
        """PID-Controller with Gain Scheduling
        
        根据当前速度动态调整PID参数：
        - 低速 (< 30 km/h): 较大的P增益，响应更灵敏
        - 中速 (30-60 km/h): 标准参数
        - 高速 (> 60 km/h): 较小的P增益，更稳定

        Args:
            error (float): Difference to the center of the detected lane.
            speed (float): Current vehicle speed in m/s.

        Returns:
            float: Steering angle to use.
        """
        # 根据速度选择增益参数
        if speed < self.SPEED_THRESHOLD_LOW:
            tau_p, tau_d, tau_i = self.GAIN_SCHEDULE['low']
        elif speed < self.SPEED_THRESHOLD_HIGH:
            tau_p, tau_d, tau_i = self.GAIN_SCHEDULE['medium']
        else:
            tau_p, tau_d, tau_i = self.GAIN_SCHEDULE['high']
        
        # 计算比例项
        p_term = -tau_p * error
        
        # 计算微分项
        if len(self.errors) > 2:
            prev_error = self.errors[-2]
        else:
            prev_error = self.errors[-1] if self.errors else error
        d_term = -tau_d * (error - prev_error) / 0.1
        
        # 计算积分项（带自适应饱和）
        if self.use_adaptive_integral:
            self.integral += error
            # 根据速度调整积分限幅
            adaptive_limit = self.integral_limit / (1 + speed * 0.1)
            self.integral = np.clip(self.integral, -adaptive_limit, adaptive_limit)
            i_term = -tau_i * self.integral
        else:
            i_term = -tau_i * sum(self.errors)
        
        steer = p_term + d_term + i_term
        return steer
    
    def _mpc_controller(self, error:float, speed:float) -> float:
        """Model Predictive Controller
        
        预测未来N步的误差，优化控制序列，选择最优控制输入。

        Args:
            error (float): Current error.
            speed (float): Current vehicle speed in m/s.

        Returns:
            float: Steering angle to use.
        """
        # 确保有足够的历史数据
        if len(self.error_history) < self.mpc_history_len:
            # 历史数据不足，使用标准PID
            return self._pid_controller(error)
        
        # MPC参数
        Q = 1.0      # 误差权重
        R = 0.1      # 控制量权重
        dt = 0.1     # 时间步长
        
        # 简化的车辆动力学模型（转向误差模型）
        # x(k+1) = x(k) + v * sin(delta) * dt
        # 其中 x 是横向误差，v 是速度，delta 是转向角
        
        # 使用二次规划求解最优控制序列
        # 最小化: sum(Q * error^2 + R * steer^2)
        
        # 简单的近似：基于预测误差趋势计算控制量
        error_array = np.array(self.error_history)
        
        # 计算误差趋势（用于预测）
        if len(error_array) >= 3:
            trend = (error_array[-1] - error_array[-3]) / 2
        else:
            trend = 0
        
        # 预测未来误差
        predicted_errors = []
        for i in range(self.mpc_horizon):
            # 误差会逐渐减小（假设车道保持有效）
            predicted_error = error + trend * (i + 1) * 0.5
            predicted_errors.append(predicted_error)
        
        # 计算最优控制序列
        # 使用简化的梯度下降法求解
        best_steer = 0
        best_cost = float('inf')
        
        # 尝试不同的控制量
        for delta in np.linspace(-1, 1, 21):
            cost = 0
            for i, pred_err in enumerate(predicted_errors):
                # 控制量随时间递减（控制时域内）
                control_weight = 1.0 if i < self.mpc_control_horizon else 0
                # 预测下一时刻的误差（考虑控制效果）
                next_err = pred_err - delta * speed * dt * 0.5
                # 成本函数：误差成本 + 控制成本
                cost += Q * next_err**2 + R * (delta * control_weight)**2
            
            if cost < best_cost:
                best_cost = cost
                best_steer = delta
        
        # 最终控制量 = 预测最优控制 + PID反馈校正
        pid_correction = -self.tau_p * error - self.tau_d * trend
        steer = best_steer * 0.7 + pid_correction * 0.3
        
        # 限制转向角度
        steer = np.clip(steer, -1.0, 1.0)
        
        return steer
