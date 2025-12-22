"""
性能分析器模块
负责监控和报告系统性能
作者: xiaoshiyuan888
优化版本：增加更多性能指标和趋势分析
"""

import time
import os
import csv
import numpy as np
from datetime import datetime
from collections import deque, Counter
import statistics


class PerformanceAnalyzer:
    """增强的性能分析器 - 监控和报告系统性能"""

    def __init__(self, speech_manager=None, psutil_lib=None, config=None):
        self.speech_manager = speech_manager
        self.psutil_lib = psutil_lib
        self.config = config
        self.start_time = time.time()
        self.session_start_time = time.time()

        # 帧率统计
        self.frame_times = deque(maxlen=500)
        self.frame_count = 0
        self.fps_history = deque(maxlen=200)
        self.frame_time_history = deque(maxlen=100)

        # 新增：FPS波动性分析
        self.fps_jitter_history = deque(maxlen=50)  # FPS波动
        self.fps_stability_score = 100  # FPS稳定性评分

        # 手势识别性能
        self.gesture_recognition_times = deque(maxlen=200)
        self.avg_recognition_time = 0
        self.max_recognition_time = 0
        self.min_recognition_time = float('inf')
        self.recognition_time_std = 0

        # 新增：手势识别延迟分布
        self.recognition_latency_distribution = {
            'excellent': 0,  # <20ms
            'good': 0,       # 20-35ms
            'fair': 0,       # 35-50ms
            'poor': 0,       # 50-100ms
            'bad': 0         # >100ms
        }

        # 系统资源监控
        self.cpu_usage_history = deque(maxlen=200)
        self.memory_usage_history = deque(maxlen=200)
        self.cpu_trend = deque(maxlen=50)
        self.memory_trend = deque(maxlen=50)

        # 新增：GPU监控（如果可用）
        self.gpu_usage_history = deque(maxlen=100)
        self.gpu_available = False
        self.gpu_memory_history = deque(maxlen=100)

        # 性能事件记录
        self.performance_events = []
        self.performance_snapshots = []
        self.performance_anomalies = []

        # 手势统计
        self.gesture_counts = {}
        self.gesture_confidence_sum = {}
        self.gesture_confidence_count = {}
        self.gesture_recognition_latency = {}

        # 新增：手势转换统计
        self.gesture_transitions = {}  # 记录手势切换频率
        self.last_gesture = None

        # 错误统计
        self.error_count = 0
        self.warning_count = 0
        self.critical_count = 0

        # 无人机控制统计
        self.drone_commands = 0
        self.successful_commands = 0
        self.failed_commands = 0
        self.command_latency_history = deque(maxlen=100)

        # 轨迹记录统计
        self.recording_sessions = 0
        self.total_trajectory_points = 0
        self.trajectory_recording_time = 0

        # 新增：系统响应时间统计
        self.system_response_times = deque(maxlen=50)
        self.avg_system_response_time = 0

        # 性能日志
        self.performance_log = []
        self.anomaly_log = []
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'performance_log.csv')
        self.anomaly_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anomaly_log.csv')

        # 性能阈值
        self.performance_thresholds = {
            'fps_excellent': 30,
            'fps_good': 20,
            'fps_warning': 15,
            'fps_critical': 5,
            'cpu_excellent': 40,
            'cpu_good': 60,
            'cpu_warning': 80,
            'cpu_critical': 90,
            'memory_excellent': 50,
            'memory_good': 70,
            'memory_warning': 85,
            'memory_critical': 95,
            'recognition_excellent': 20,
            'recognition_good': 35,
            'recognition_warning': 50,
            'recognition_critical': 100,
            'frame_time_excellent': 33,
            'frame_time_warning': 66,
            'frame_time_critical': 200,
        }

        # 性能状态
        self.performance_status = "优秀"
        self.performance_score = 100
        self.last_performance_report = 0
        self.auto_report_interval = 60

        # 趋势分析
        self.fps_trend = "稳定"
        self.cpu_trend = "稳定"
        self.memory_trend = "稳定"

        # 新增：性能预测
        self.fps_prediction = 0
        self.cpu_prediction = 0
        self.memory_prediction = 0

        # 异常检测
        self.anomaly_detection_enabled = True
        self.last_anomaly_check = 0
        self.anomaly_check_interval = 10

        # 初始化GPU监控
        self.init_gpu_monitoring()

        print("✓ 增强的性能分析器已初始化")

    def init_gpu_monitoring(self):
        """初始化GPU监控"""
        try:
            # 尝试导入GPU监控库
            import GPUtil
            self.gpu_available = True
            self.gpu_lib = GPUtil
            print("[GPU] ✓ GPU监控库就绪")
        except ImportError:
            print("[GPU] ⚠ GPU监控库未找到，GPU监控功能受限")
            self.gpu_available = False

    def update_gpu_stats(self):
        """更新GPU统计"""
        if not self.gpu_available:
            return

        try:
            gpus = self.gpu_lib.getGPUs()
            if gpus:
                gpu = gpus[0]  # 使用第一个GPU
                self.gpu_usage_history.append(gpu.load * 100)
                self.gpu_memory_history.append(gpu.memoryUtil * 100)
        except:
            pass

    def update_frame(self):
        """更新帧统计"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_count += 1

        # 计算当前FPS
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                current_fps = (len(self.frame_times) - 1) / time_span
                self.fps_history.append(current_fps)

                # 计算帧时间（毫秒）
                if len(self.frame_times) >= 2:
                    frame_time = (self.frame_times[-1] - self.frame_times[-2]) * 1000
                    self.frame_time_history.append(frame_time)

                    # 计算FPS波动
                    if len(self.fps_history) >= 2:
                        fps_jitter = abs(self.fps_history[-1] - self.fps_history[-2])
                        self.fps_jitter_history.append(fps_jitter)

        # 更新GPU统计
        if self.frame_count % 30 == 0:  # 每30帧更新一次GPU信息
            self.update_gpu_stats()

        # 更新性能评分
        self.update_performance_score()

        # 更新系统响应时间
        if len(self.system_response_times) > 0:
            self.avg_system_response_time = statistics.mean(list(self.system_response_times))

    def update_performance_score(self):
        """更新性能评分"""
        score = 100

        # 基于FPS评分
        if len(self.fps_history) > 0:
            avg_fps = self.get_average_fps()
            if avg_fps >= self.performance_thresholds['fps_excellent']:
                score -= 0
            elif avg_fps >= self.performance_thresholds['fps_good']:
                score -= 10
            elif avg_fps >= self.performance_thresholds['fps_warning']:
                score -= 25
            elif avg_fps >= self.performance_thresholds['fps_critical']:
                score -= 50
            else:
                score -= 70

        # 基于FPS稳定性评分
        if len(self.fps_jitter_history) > 0:
            avg_jitter = statistics.mean(list(self.fps_jitter_history))
            if avg_jitter < 2:
                score -= 0
            elif avg_jitter < 5:
                score -= 5
            elif avg_jitter < 10:
                score -= 15
            else:
                score -= 25
            self.fps_stability_score = max(0, 100 - avg_jitter * 10)

        # 基于CPU评分
        cpu_usage = self.get_current_cpu_usage()
        if cpu_usage <= self.performance_thresholds['cpu_excellent']:
            score -= 0
        elif cpu_usage <= self.performance_thresholds['cpu_good']:
            score -= 5
        elif cpu_usage <= self.performance_thresholds['cpu_warning']:
            score -= 15
        elif cpu_usage <= self.performance_thresholds['cpu_critical']:
            score -= 30
        else:
            score -= 50

        # 基于内存评分
        memory_usage = self.get_current_memory_usage()
        if memory_usage <= self.performance_thresholds['memory_excellent']:
            score -= 0
        elif memory_usage <= self.performance_thresholds['memory_good']:
            score -= 5
        elif memory_usage <= self.performance_thresholds['memory_warning']:
            score -= 15
        elif memory_usage <= self.performance_thresholds['memory_critical']:
            score -= 30
        else:
            score -= 50

        # 基于识别时间评分
        if self.avg_recognition_time > 0:
            if self.avg_recognition_time <= self.performance_thresholds['recognition_excellent']:
                score -= 0
            elif self.avg_recognition_time <= self.performance_thresholds['recognition_good']:
                score -= 5
            elif self.avg_recognition_time <= self.performance_thresholds['recognition_warning']:
                score -= 15
            elif self.avg_recognition_time <= self.performance_thresholds['recognition_critical']:
                score -= 30
            else:
                score -= 50

        # 基于GPU评分（如果可用）
        if self.gpu_available and len(self.gpu_usage_history) > 0:
            gpu_usage = self.gpu_usage_history[-1] if self.gpu_usage_history else 0
            if gpu_usage < 60:
                score -= 0
            elif gpu_usage < 80:
                score -= 5
            elif gpu_usage < 90:
                score -= 10
            else:
                score -= 20

        self.performance_score = max(0, min(100, score))

        # 更新性能状态
        if self.performance_score >= 90:
            self.performance_status = "优秀"
        elif self.performance_score >= 70:
            self.performance_status = "良好"
        elif self.performance_score >= 50:
            self.performance_status = "一般"
        elif self.performance_score >= 30:
            self.performance_status = "警告"
        else:
            self.performance_status = "严重"

    def update_gesture_recognition_time(self, recognition_time_ms):
        """更新手势识别时间"""
        self.gesture_recognition_times.append(recognition_time_ms)

        # 更新识别时间统计
        if len(self.gesture_recognition_times) > 0:
            times_list = list(self.gesture_recognition_times)
            self.avg_recognition_time = np.mean(times_list)
            self.max_recognition_time = max(self.max_recognition_time, recognition_time_ms)
            self.min_recognition_time = min(self.min_recognition_time, recognition_time_ms)

            if len(times_list) >= 2:
                self.recognition_time_std = np.std(times_list)

        # 更新识别延迟分布
        if recognition_time_ms < 20:
            self.recognition_latency_distribution['excellent'] += 1
        elif recognition_time_ms < 35:
            self.recognition_latency_distribution['good'] += 1
        elif recognition_time_ms < 50:
            self.recognition_latency_distribution['fair'] += 1
        elif recognition_time_ms < 100:
            self.recognition_latency_distribution['poor'] += 1
        else:
            self.recognition_latency_distribution['bad'] += 1

    def update_system_resources(self):
        """更新系统资源使用情况"""
        try:
            if self.psutil_lib:
                cpu_percent = self.psutil_lib.cpu_percent(interval=0.1)
                memory_percent = self.psutil_lib.virtual_memory().percent

                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory_percent)

                # 更新趋势
                if len(self.cpu_usage_history) >= 10:
                    recent_cpu = list(self.cpu_usage_history)[-10:]
                    self.cpu_trend.append(statistics.mean(recent_cpu))

                if len(self.memory_usage_history) >= 10:
                    recent_memory = list(self.memory_usage_history)[-10:]
                    self.memory_trend.append(statistics.mean(recent_memory))

                # 性能预测
                self.predict_performance()

                # 检查性能问题和异常
                self.check_performance_issues(cpu_percent, memory_percent)
                self.detect_anomalies()
        except:
            pass

    def predict_performance(self):
        """性能预测"""
        try:
            # 简单的线性回归预测
            if len(self.fps_history) >= 10:
                recent_fps = list(self.fps_history)[-10:]
                if len(recent_fps) >= 2:
                    # 计算趋势
                    x = np.arange(len(recent_fps))
                    y = np.array(recent_fps)
                    z = np.polyfit(x, y, 1)
                    self.fps_prediction = max(0, z[0] * len(recent_fps) + z[1])

            if len(self.cpu_usage_history) >= 10:
                recent_cpu = list(self.cpu_usage_history)[-10:]
                if len(recent_cpu) >= 2:
                    x = np.arange(len(recent_cpu))
                    y = np.array(recent_cpu)
                    z = np.polyfit(x, y, 1)
                    self.cpu_prediction = min(100, max(0, z[0] * len(recent_cpu) + z[1]))

            if len(self.memory_usage_history) >= 10:
                recent_memory = list(self.memory_usage_history)[-10:]
                if len(recent_memory) >= 2:
                    x = np.arange(len(recent_memory))
                    y = np.array(recent_memory)
                    z = np.polyfit(x, y, 1)
                    self.memory_prediction = min(100, max(0, z[0] * len(recent_memory) + z[1]))
        except:
            pass

    def detect_anomalies(self):
        """检测性能异常"""
        if not self.anomaly_detection_enabled:
            return

        current_time = time.time()
        if current_time - self.last_anomaly_check < self.anomaly_check_interval:
            return

        self.last_anomaly_check = current_time

        anomalies = []

        # 检测FPS异常
        if len(self.fps_history) >= 10:
            recent_fps = list(self.fps_history)[-10:]
            avg_fps = statistics.mean(recent_fps)
            std_fps = statistics.stdev(recent_fps) if len(recent_fps) > 1 else 0

            # 如果FPS突然下降超过50%
            if len(self.fps_history) >= 20:
                older_fps = list(self.fps_history)[-20:-10]
                if len(older_fps) > 0:
                    older_avg = statistics.mean(older_fps)
                    if older_avg > 0 and avg_fps / older_avg < 0.5:
                        anomalies.append(("FPS骤降", f"FPS从{older_avg:.1f}降至{avg_fps:.1f}"))

            # 检测FPS波动过大
            if std_fps > 10:
                anomalies.append(("FPS不稳定", f"FPS波动过大: 标准差{std_fps:.1f}"))

        # 检测CPU使用率异常
        if len(self.cpu_usage_history) >= 10:
            recent_cpu = list(self.cpu_usage_history)[-10:]
            avg_cpu = statistics.mean(recent_cpu)

            # CPU使用率突然飙升
            if len(self.cpu_usage_history) >= 20:
                older_cpu = list(self.cpu_usage_history)[-20:-10]
                if len(older_cpu) > 0:
                    older_avg = statistics.mean(older_cpu)
                    if avg_cpu - older_avg > 30:  # 突然增加30%以上
                        anomalies.append(("CPU飙升", f"CPU从{older_avg:.1f}%升至{avg_cpu:.1f}%"))

        # 检测内存泄漏迹象
        if len(self.memory_usage_history) >= 30:
            memory_values = list(self.memory_usage_history)
            # 检查内存是否持续增长
            if len(memory_values) >= 30:
                first_half = memory_values[:15]
                second_half = memory_values[15:]
                if len(first_half) > 0 and len(second_half) > 0:
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    if second_avg - first_avg > 10:  # 内存增长超过10%
                        anomalies.append(("内存增长", f"内存从{first_avg:.1f}%增长到{second_avg:.1f}%"))

        # 检测手势识别时间异常
        if len(self.gesture_recognition_times) >= 10:
            recent_times = list(self.gesture_recognition_times)[-10:]
            avg_time = statistics.mean(recent_times)
            if avg_time > 80:  # 识别时间超过80ms
                anomalies.append(("识别缓慢", f"手势识别平均时间{avg_time:.1f}ms"))

        # 记录异常
        for anomaly_type, message in anomalies:
            anomaly = {
                'timestamp': current_time,
                'type': anomaly_type,
                'message': message,
                'fps': self.get_current_fps(),
                'cpu': self.get_current_cpu_usage(),
                'memory': self.get_current_memory_usage(),
                'recognition_time': self.avg_recognition_time,
                'performance_score': self.performance_score
            }
            self.performance_anomalies.append(anomaly)
            self.log_anomaly(anomaly)

            # 语音提示严重异常
            if (self.speech_manager and
                self.speech_manager.enabled and
                ("骤降" in anomaly_type or "飙升" in anomaly_type or "缓慢" in anomaly_type)):
                self.speech_manager.speak_direct(f"检测到性能{anomaly_type}")

    def check_performance_issues(self, cpu_percent, memory_percent):
        """检查性能问题"""
        issues = []

        # 检查FPS
        if len(self.fps_history) > 0:
            avg_fps = self.get_average_fps()
            current_fps = self.get_current_fps()

            if current_fps < self.performance_thresholds['fps_critical']:
                issues.append(("严重", f"帧率极低: {current_fps:.1f} FPS"))
                self.performance_status = "严重"
                self.critical_count += 1
            elif current_fps < self.performance_thresholds['fps_warning']:
                issues.append(("警告", f"帧率较低: {current_fps:.1f} FPS"))
                if self.performance_status == "优秀" or self.performance_status == "良好":
                    self.performance_status = "警告"
                self.warning_count += 1

        # 检查CPU使用率
        if cpu_percent > self.performance_thresholds['cpu_critical']:
            issues.append(("严重", f"CPU使用率极高: {cpu_percent:.1f}%"))
            self.performance_status = "严重"
            self.critical_count += 1
        elif cpu_percent > self.performance_thresholds['cpu_warning']:
            issues.append(("警告", f"CPU使用率较高: {cpu_percent:.1f}%"))
            if self.performance_status == "优秀" or self.performance_status == "良好":
                self.performance_status = "警告"
            self.warning_count += 1

        # 检查内存使用率
        if memory_percent > self.performance_thresholds['memory_critical']:
            issues.append(("严重", f"内存使用率极高: {memory_percent:.1f}%"))
            self.performance_status = "严重"
            self.critical_count += 1
        elif memory_percent > self.performance_thresholds['memory_warning']:
            issues.append(("警告", f"内存使用率较高: {memory_percent:.1f}%"))
            if self.performance_status == "优秀" or self.performance_status == "良好":
                self.performance_status = "警告"
            self.warning_count += 1

        # 检查手势识别时间
        if self.avg_recognition_time > self.performance_thresholds['recognition_critical']:
            issues.append(("严重", f"手势识别时间极长: {self.avg_recognition_time:.1f}ms"))
            self.performance_status = "严重"
            self.critical_count += 1
        elif self.avg_recognition_time > self.performance_thresholds['recognition_warning']:
            issues.append(("警告", f"手势识别时间较长: {self.avg_recognition_time:.1f}ms"))
            if self.performance_status == "优秀" or self.performance_status == "良好":
                self.performance_status = "警告"
            self.warning_count += 1

        # 检查帧时间
        if len(self.frame_time_history) > 0:
            avg_frame_time = statistics.mean(list(self.frame_time_history)) if self.frame_time_history else 0
            if avg_frame_time > self.performance_thresholds['frame_time_critical']:
                issues.append(("严重", f"帧处理时间极长: {avg_frame_time:.1f}ms"))
                self.performance_status = "严重"
                self.critical_count += 1
            elif avg_frame_time > self.performance_thresholds['frame_time_warning']:
                issues.append(("警告", f"帧处理时间较长: {avg_frame_time:.1f}ms"))
                if self.performance_status == "优秀" or self.performance_status == "良好":
                    self.performance_status = "警告"
                self.warning_count += 1

        # 检查GPU使用率
        if self.gpu_available and len(self.gpu_usage_history) > 0:
            gpu_usage = self.gpu_usage_history[-1] if self.gpu_usage_history else 0
            if gpu_usage > 95:
                issues.append(("严重", f"GPU使用率极高: {gpu_usage:.1f}%"))
                self.performance_status = "严重"
                self.critical_count += 1
            elif gpu_usage > 90:
                issues.append(("警告", f"GPU使用率较高: {gpu_usage:.1f}%"))
                if self.performance_status == "优秀" or self.performance_status == "良好":
                    self.performance_status = "警告"
                self.warning_count += 1

        # 记录性能事件
        if issues:
            for level, message in issues:
                self.add_performance_event(level, message)

                # 语音提示（仅在状态变化时）
                if (self.speech_manager and
                        self.speech_manager.enabled and
                        level == "严重"):
                    current_time = time.time()
                    if current_time - self.last_performance_report > 10:
                        self.speech_manager.speak_direct(f"性能{level}: {message}")
                        self.last_performance_report = current_time

    def add_performance_event(self, level, message):
        """添加性能事件"""
        event = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            'session_time': time.time() - self.session_start_time,
            'fps': self.get_current_fps(),
            'cpu': self.get_current_cpu_usage(),
            'memory': self.get_current_memory_usage(),
            'recognition_time': self.avg_recognition_time,
            'performance_score': self.performance_score
        }
        self.performance_events.append(event)

        # 记录到日志
        self.log_performance_event(event)

        if level == "警告":
            self.warning_count += 1
        elif level == "严重":
            self.error_count += 1

    def log_performance_event(self, event):
        """记录性能事件到日志"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
            'session_time': f"{event['session_time']:.1f}",
            'level': event['level'],
            'message': event['message'],
            'fps': f"{event['fps']:.1f}",
            'cpu': f"{event['cpu']:.1f}",
            'memory': f"{event['memory']:.1f}",
            'recognition_time': f"{event['recognition_time']:.1f}",
            'performance_score': f"{event['performance_score']:.1f}"
        }
        self.performance_log.append(log_entry)

    def log_anomaly(self, anomaly):
        """记录异常到日志"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(anomaly['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
            'type': anomaly['type'],
            'message': anomaly['message'],
            'fps': f"{anomaly['fps']:.1f}",
            'cpu': f"{anomaly['cpu']:.1f}",
            'memory': f"{anomaly['memory']:.1f}",
            'recognition_time': f"{anomaly['recognition_time']:.1f}",
            'performance_score': f"{anomaly['performance_score']:.1f}"
        }
        self.anomaly_log.append(log_entry)

    def record_gesture(self, gesture, confidence):
        """记录手势统计"""
        if gesture not in self.gesture_counts:
            self.gesture_counts[gesture] = 0
            self.gesture_confidence_sum[gesture] = 0
            self.gesture_confidence_count[gesture] = 0
            self.gesture_recognition_latency[gesture] = []

        self.gesture_counts[gesture] += 1
        self.gesture_confidence_sum[gesture] += confidence
        self.gesture_confidence_count[gesture] += 1

        # 记录手势转换
        if self.last_gesture is not None and self.last_gesture != gesture:
            transition_key = f"{self.last_gesture}->{gesture}"
            if transition_key not in self.gesture_transitions:
                self.gesture_transitions[transition_key] = 0
            self.gesture_transitions[transition_key] += 1

        self.last_gesture = gesture

        # 记录最近一次识别时间
        if len(self.gesture_recognition_times) > 0:
            last_time = self.gesture_recognition_times[-1]
            self.gesture_recognition_latency[gesture].append(last_time)
            # 只保留最近20个
            if len(self.gesture_recognition_latency[gesture]) > 20:
                self.gesture_recognition_latency[gesture].pop(0)

    def record_drone_command(self, success=True, latency=0):
        """记录无人机命令"""
        self.drone_commands += 1
        if success:
            self.successful_commands += 1
        else:
            self.failed_commands += 1

        if latency > 0:
            self.command_latency_history.append(latency)

    def record_recording_session(self, points_count=0):
        """记录录制会话"""
        self.recording_sessions += 1
        self.total_trajectory_points += points_count
        if points_count > 0:
            self.trajectory_recording_time = time.time() - self.session_start_time

    def record_system_response_time(self, response_time):
        """记录系统响应时间"""
        self.system_response_times.append(response_time)

    def take_snapshot(self, label=""):
        """拍摄性能快照"""
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'fps': self.get_current_fps(),
            'avg_fps': self.get_average_fps(),
            'fps_history': list(self.fps_history)[-20:] if len(self.fps_history) >= 20 else list(self.fps_history),
            'avg_recognition_time': self.avg_recognition_time,
            'max_recognition_time': self.max_recognition_time,
            'min_recognition_time': self.min_recognition_time if self.min_recognition_time != float('inf') else 0,
            'recognition_time_std': self.recognition_time_std,
            'cpu_usage': self.get_current_cpu_usage(),
            'memory_usage': self.get_current_memory_usage(),
            'cpu_history': list(self.cpu_usage_history)[-20:] if len(self.cpu_usage_history) >= 20 else list(self.cpu_usage_history),
            'memory_history': list(self.memory_usage_history)[-20:] if len(self.memory_usage_history) >= 20 else list(self.memory_usage_history),
            'gesture_counts': dict(self.gesture_counts),
            'performance_status': self.performance_status,
            'performance_score': self.performance_score,
            'frame_count': self.frame_count,
            'session_duration': time.time() - self.session_start_time,
            'warning_count': self.warning_count,
            'error_count': self.error_count,
            'critical_count': self.critical_count,
            'fps_stability_score': self.fps_stability_score,
            'recognition_latency_distribution': dict(self.recognition_latency_distribution),
            'gesture_transitions': dict(self.gesture_transitions),
            'avg_system_response_time': self.avg_system_response_time
        }

        # 添加GPU信息（如果可用）
        if self.gpu_available and len(self.gpu_usage_history) > 0:
            snapshot['gpu_usage'] = self.gpu_usage_history[-1] if self.gpu_usage_history else 0
            snapshot['gpu_memory'] = self.gpu_memory_history[-1] if self.gpu_memory_history else 0

        self.performance_snapshots.append(snapshot)

        print(f"📸 性能快照已保存: {label}")
        return snapshot

    def get_current_fps(self):
        """获取当前FPS"""
        if len(self.fps_history) > 0:
            return self.fps_history[-1]
        return 0

    def get_average_fps(self):
        """获取平均FPS"""
        if len(self.fps_history) > 0:
            return np.mean(list(self.fps_history))
        return 0

    def get_fps_percentile(self, percentile):
        """获取FPS百分位数"""
        if len(self.fps_history) >= 10:
            fps_values = list(self.fps_history)
            return np.percentile(fps_values, percentile)
        return 0

    def get_current_cpu_usage(self):
        """获取当前CPU使用率"""
        if len(self.cpu_usage_history) > 0:
            return self.cpu_usage_history[-1]
        return 0

    def get_current_memory_usage(self):
        """获取当前内存使用率"""
        if len(self.memory_usage_history) > 0:
            return self.memory_usage_history[-1]
        return 0

    def get_cpu_trend(self):
        """获取CPU使用趋势"""
        if len(self.cpu_trend) >= 3:
            recent = list(self.cpu_trend)[-3:]
            if recent[-1] > recent[0] + 5:
                return "上升"
            elif recent[-1] < recent[0] - 5:
                return "下降"
        return "稳定"

    def get_memory_trend(self):
        """获取内存使用趋势"""
        if len(self.memory_trend) >= 3:
            recent = list(self.memory_trend)[-3:]
            if recent[-1] > recent[0] + 3:
                return "上升"
            elif recent[-1] < recent[0] - 3:
                return "下降"
        return "稳定"

    def get_fps_trend(self):
        """获取FPS趋势"""
        if len(self.fps_history) >= 10:
            recent = list(self.fps_history)[-10:]
            older = list(self.fps_history)[-20:-10] if len(self.fps_history) >= 20 else recent[:5]

            if len(recent) > 0 and len(older) > 0:
                recent_avg = statistics.mean(recent)
                older_avg = statistics.mean(older)

                if recent_avg > older_avg + 5:
                    return "上升"
                elif recent_avg < older_avg - 5:
                    return "下降"
        return "稳定"

    def get_fps_stability(self):
        """获取FPS稳定性评分"""
        return self.fps_stability_score

    def get_gpu_usage(self):
        """获取GPU使用率"""
        if self.gpu_available and len(self.gpu_usage_history) > 0:
            return self.gpu_usage_history[-1]
        return 0

    def generate_report(self, detailed=True):
        """生成性能报告"""
        report_time = time.time()
        session_duration = report_time - self.session_start_time

        # 基础报告
        report = {
            '生成时间': datetime.fromtimestamp(report_time).strftime('%Y-%m-%d %H:%M:%S'),
            '会话时长': f"{session_duration:.1f}秒",
            '总帧数': self.frame_count,
            '平均FPS': f"{self.get_average_fps():.1f}",
            '当前FPS': f"{self.get_current_fps():.1f}",
            '最低FPS': f"{min(self.fps_history) if self.fps_history else 0:.1f}",
            'FPS稳定性': f"{self.get_fps_percentile(90) - self.get_fps_percentile(10):.1f}",
            'FPS稳定性评分': f"{self.fps_stability_score:.0f}",
            '平均手势识别时间': f"{self.avg_recognition_time:.1f}ms",
            '最快识别时间': f"{self.min_recognition_time if self.min_recognition_time != float('inf') else 0:.1f}ms",
            '最慢识别时间': f"{self.max_recognition_time:.1f}ms",
            '识别时间标准差': f"{self.recognition_time_std:.1f}ms",
            '当前CPU使用率': f"{self.get_current_cpu_usage():.1f}%",
            '当前内存使用率': f"{self.get_current_memory_usage():.1f}%",
            '性能状态': self.performance_status,
            '性能评分': f"{self.performance_score:.0f}",
            '警告数量': self.warning_count,
            '错误数量': self.error_count,
            '严重问题数量': self.critical_count,
            '无人机命令': {
                '总数': self.drone_commands,
                '成功': self.successful_commands,
                '失败': self.failed_commands,
                '成功率': f"{(self.successful_commands / self.drone_commands * 100 if self.drone_commands > 0 else 0):.1f}%",
                '平均延迟': f"{statistics.mean(self.command_latency_history) if self.command_latency_history else 0:.1f}ms"
            },
            '录制统计': {
                '会话数': self.recording_sessions,
                '总轨迹点数': self.total_trajectory_points,
                '录制时长': f"{self.trajectory_recording_time:.1f}秒"
            },
            '趋势分析': {
                'FPS趋势': self.get_fps_trend(),
                'CPU趋势': self.get_cpu_trend(),
                '内存趋势': self.get_memory_trend(),
                'FPS预测': f"{self.fps_prediction:.1f}",
                'CPU预测': f"{self.cpu_prediction:.1f}%",
                '内存预测': f"{self.memory_prediction:.1f}%"
            },
            '系统响应时间': f"{self.avg_system_response_time:.1f}ms"
        }

        # 添加GPU信息（如果可用）
        if self.gpu_available:
            report['GPU使用率'] = f"{self.get_gpu_usage():.1f}%"

        # 详细报告
        if detailed:
            # 手势统计
            gesture_stats = {}
            for gesture in self.gesture_counts:
                count = self.gesture_counts[gesture]
                if gesture in self.gesture_confidence_count and self.gesture_confidence_count[gesture] > 0:
                    avg_confidence = self.gesture_confidence_sum[gesture] / self.gesture_confidence_count[gesture]
                else:
                    avg_confidence = 0

                # 计算手势识别延迟
                avg_latency = 0
                if gesture in self.gesture_recognition_latency and self.gesture_recognition_latency[gesture]:
                    avg_latency = statistics.mean(self.gesture_recognition_latency[gesture])

                gesture_stats[gesture] = {
                    '次数': count,
                    '占比': f"{(count / self.frame_count * 100 if self.frame_count > 0 else 0):.1f}%",
                    '平均置信度': f"{avg_confidence:.2f}",
                    '平均延迟': f"{avg_latency:.1f}ms"
                }

            report['手势统计'] = gesture_stats

            # 手势转换统计
            if self.gesture_transitions:
                report['手势转换统计'] = dict(sorted(
                    self.gesture_transitions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])  # 只显示前10个最常见转换

            # 识别延迟分布
            total_latency = sum(self.recognition_latency_distribution.values())
            if total_latency > 0:
                latency_dist = {}
                for category, count in self.recognition_latency_distribution.items():
                    percentage = (count / total_latency * 100) if total_latency > 0 else 0
                    latency_dist[category] = f"{count}次({percentage:.1f}%)"
                report['识别延迟分布'] = latency_dist

            # 性能事件
            if self.performance_events:
                recent_events = list(self.performance_events)[-10:]
                report['最近性能事件'] = [
                    {
                        '时间': datetime.fromtimestamp(e['timestamp']).strftime('%H:%M:%S'),
                        '级别': e['level'],
                        '消息': e['message'],
                        'FPS': f"{e['fps']:.1f}",
                        'CPU': f"{e['cpu']:.1f}%",
                        '内存': f"{e['memory']:.1f}%"
                    }
                    for e in recent_events
                ]

            # 性能异常
            if self.performance_anomalies:
                recent_anomalies = list(self.performance_anomalies)[-5:]
                report['最近性能异常'] = [
                    {
                        '时间': datetime.fromtimestamp(a['timestamp']).strftime('%H:%M:%S'),
                        '类型': a['type'],
                        '消息': a['message']
                    }
                    for a in recent_anomalies
                ]

            # 性能快照
            if self.performance_snapshots:
                report['性能快照数'] = len(self.performance_snapshots)
                report['最近快照时间'] = datetime.fromtimestamp(self.performance_snapshots[-1]['timestamp']).strftime('%H:%M:%S')

            # 系统建议
            suggestions = self.generate_suggestions()
            if suggestions:
                report['优化建议'] = suggestions

            # 性能分布
            if len(self.fps_history) >= 20:
                fps_values = list(self.fps_history)
                report['FPS分布'] = {
                    'P10': f"{np.percentile(fps_values, 10):.1f}",
                    'P50': f"{np.percentile(fps_values, 50):.1f}",
                    'P90': f"{np.percentile(fps_values, 90):.1f}",
                    'P95': f"{np.percentile(fps_values, 95):.1f}",
                    '标准差': f"{np.std(fps_values):.1f}"
                }

        return report

    def generate_suggestions(self):
        """生成优化建议"""
        suggestions = []

        # 检查FPS
        avg_fps = self.get_average_fps()
        if avg_fps < self.performance_thresholds['fps_warning']:
            suggestions.append(f"帧率较低({avg_fps:.1f}FPS)，建议切换到'最快'性能模式，或降低摄像头分辨率")
            if self.performance_score < 50:
                suggestions.append("考虑升级硬件配置（CPU/GPU）以提升性能")

        # 检查FPS稳定性
        if len(self.fps_jitter_history) > 0:
            avg_jitter = statistics.mean(list(self.fps_jitter_history))
            if avg_jitter > 5:
                suggestions.append(f"FPS波动较大({avg_jitter:.1f}FPS)，建议关闭其他运行程序，优化系统性能")

        # 检查CPU
        cpu_usage = self.get_current_cpu_usage()
        if cpu_usage > self.performance_thresholds['cpu_warning']:
            suggestions.append(f"CPU使用率较高({cpu_usage:.1f}%)，请关闭其他占用CPU的程序")
            if cpu_usage > 90:
                suggestions.append("考虑优化手势识别算法或使用硬件加速")

        # 检查内存
        memory_usage = self.get_current_memory_usage()
        if memory_usage > self.performance_thresholds['memory_warning']:
            suggestions.append(f"内存使用率较高({memory_usage:.1f}%)，请关闭不必要的程序")
            if memory_usage > 90:
                suggestions.append("考虑增加系统内存或优化内存使用")

        # 检查识别时间
        if self.avg_recognition_time > self.performance_thresholds['recognition_warning']:
            suggestions.append(f"手势识别时间较长({self.avg_recognition_time:.1f}ms)，建议调整摄像头位置或光线")
            if self.avg_recognition_time > 80:
                suggestions.append("考虑使用更简单的手势识别算法或优化当前算法")

        # 检查GPU使用率（如果可用）
        if self.gpu_available and len(self.gpu_usage_history) > 0:
            gpu_usage = self.get_gpu_usage()
            if gpu_usage > 90:
                suggestions.append(f"GPU使用率较高({gpu_usage:.1f}%)，可能影响系统性能")

        # 检查性能趋势
        if self.get_fps_trend() == "下降":
            suggestions.append("FPS呈下降趋势，建议重启程序或检查系统资源")

        # 检查手势识别延迟分布
        total_latency = sum(self.recognition_latency_distribution.values())
        if total_latency > 0:
            poor_ratio = (self.recognition_latency_distribution['poor'] + self.recognition_latency_distribution['bad']) / total_latency
            if poor_ratio > 0.3:  # 超过30%的识别时间较差
                suggestions.append(f"手势识别延迟较高，{poor_ratio*100:.0f}%的识别时间超过50ms")

        return suggestions

    def print_report(self, detailed=True):
        """打印性能报告"""
        report = self.generate_report(detailed)

        print("\n" + "=" * 100)
        print("📊 增强性能分析报告")
        print("=" * 100)

        # 基础信息
        print(f"生成时间: {report['生成时间']}")
        print(f"会话时长: {report['会话时长']}")
        print(f"总帧数: {report['总帧数']}")
        print(f"平均FPS: {report['平均FPS']}")
        print(f"当前FPS: {report['当前FPS']}")
        print(f"最低FPS: {report['最低FPS']}")
        print(f"FPS稳定性: {report['FPS稳定性']}")
        print(f"FPS稳定性评分: {report['FPS稳定性评分']}")
        print(f"平均手势识别时间: {report['平均手势识别时间']}")
        print(f"最快识别时间: {report['最快识别时间']}")
        print(f"最慢识别时间: {report['最慢识别时间']}")
        print(f"识别时间标准差: {report['识别时间标准差']}")
        print(f"当前CPU使用率: {report['当前CPU使用率']}")
        print(f"当前内存使用率: {report['当前内存使用率']}")
        print(f"性能状态: {report['性能状态']}")
        print(f"性能评分: {report['性能评分']}")
        print(f"警告数量: {report['警告数量']}")
        print(f"错误数量: {report['错误数量']}")
        print(f"严重问题数量: {report['严重问题数量']}")

        # GPU信息
        if 'GPU使用率' in report:
            print(f"GPU使用率: {report['GPU使用率']}")

        # 趋势分析
        trends = report['趋势分析']
        print(f"\n趋势分析:")
        print(f"  FPS趋势: {trends['FPS趋势']}")
        print(f"  CPU趋势: {trends['CPU趋势']}")
        print(f"  内存趋势: {trends['内存趋势']}")
        if 'FPS预测' in trends:
            print(f"  FPS预测: {trends['FPS预测']} FPS")
            print(f"  CPU预测: {trends['CPU预测']}")
            print(f"  内存预测: {trends['内存预测']}")

        # 系统响应时间
        print(f"系统响应时间: {report['系统响应时间']}")

        # 无人机命令统计
        cmd_stats = report['无人机命令']
        print(f"\n无人机命令统计:")
        print(f"  总数: {cmd_stats['总数']}")
        print(f"  成功: {cmd_stats['成功']}")
        print(f"  失败: {cmd_stats['失败']}")
        print(f"  成功率: {cmd_stats['成功率']}")
        print(f"  平均延迟: {cmd_stats['平均延迟']}")

        # 录制统计
        rec_stats = report['录制统计']
        print(f"\n录制统计:")
        print(f"  会话数: {rec_stats['会话数']}")
        print(f"  总轨迹点数: {rec_stats['总轨迹点数']}")
        print(f"  录制时长: {rec_stats['录制时长']}")

        # 详细报告
        if detailed and '手势统计' in report:
            print(f"\n手势统计:")
            for gesture, stats in report['手势统计'].items():
                print(f"  {gesture}: {stats['次数']}次 ({stats['占比']}), 平均置信度: {stats['平均置信度']}, 平均延迟: {stats['平均延迟']}")

        # 手势转换统计
        if detailed and '手势转换统计' in report and report['手势转换统计']:
            print(f"\n最常见的手势转换:")
            for transition, count in report['手势转换统计'].items():
                print(f"  {transition}: {count}次")

        # 识别延迟分布
        if detailed and '识别延迟分布' in report and report['识别延迟分布']:
            print(f"\n识别延迟分布:")
            for category, dist in report['识别延迟分布'].items():
                print(f"  {category}: {dist}")

        # FPS分布
        if detailed and 'FPS分布' in report:
            fps_dist = report['FPS分布']
            print(f"\nFPS分布:")
            print(f"  P10: {fps_dist['P10']} FPS, P50: {fps_dist['P50']} FPS, P90: {fps_dist['P90']} FPS")
            print(f"  P95: {fps_dist['P95']} FPS, 标准差: {fps_dist['标准差']} FPS")

        # 性能事件
        if detailed and '最近性能事件' in report and report['最近性能事件']:
            print(f"\n最近性能事件:")
            for event in report['最近性能事件']:
                print(f"  [{event['时间']}] {event['级别']}: {event['消息']} (FPS:{event['FPS']}, CPU:{event['CPU']}, 内存:{event['内存']})")

        # 性能异常
        if detailed and '最近性能异常' in report and report['最近性能异常']:
            print(f"\n最近性能异常:")
            for anomaly in report['最近性能异常']:
                print(f"  [{anomaly['时间']}] {anomaly['类型']}: {anomaly['消息']}")

        # 优化建议
        if detailed and '优化建议' in report and report['优化建议']:
            print(f"\n优化建议:")
            for i, suggestion in enumerate(report['优化建议'], 1):
                print(f"  {i}. {suggestion}")

        print("=" * 100)

        # 语音播报摘要
        if self.speech_manager and self.speech_manager.enabled:
            summary = (f"性能报告: 平均帧率{report['平均FPS']}，识别时间{report['平均手势识别时间']}，"
                       f"性能状态{report['性能状态']}，评分{report['性能评分']}")
            self.speech_manager.speak_direct(summary)

    def export_log(self, filename=None):
        """导出性能日志"""
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(current_dir, f'performance_log_{timestamp}.csv')

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if self.performance_log:
                    fieldnames = self.performance_log[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.performance_log)

            print(f"📤 性能日志已导出到: {filename}")
            return True
        except Exception as e:
            print(f"❌ 导出性能日志失败: {e}")
            return False

    def export_anomaly_log(self, filename=None):
        """导出异常日志"""
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(current_dir, f'anomaly_log_{timestamp}.csv')

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if self.anomaly_log:
                    fieldnames = self.anomaly_log[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.anomaly_log)

            print(f"📤 异常日志已导出到: {filename}")
            return True
        except Exception as e:
            print(f"❌ 导出异常日志失败: {e}")
            return False

    def auto_report(self):
        """自动性能报告（定期执行）"""
        current_time = time.time()
        if current_time - self.last_performance_report > self.auto_report_interval:
            # 生成简要报告
            report = self.generate_report(detailed=False)

            # 检查是否需要报告
            if (self.performance_status == "严重" or
                    self.warning_count > 5 or
                    self.error_count > 0 or
                    self.performance_score < 40):

                print(f"⚠ 自动性能检查: {report['性能状态']}({report['性能评分']}分), "
                      f"FPS: {report['当前FPS']}, CPU: {report['当前CPU使用率']}, "
                      f"内存: {report['当前内存使用率']}, 识别时间: {report['平均手势识别时间']}")

                # 语音提示
                if (self.speech_manager and
                        self.speech_manager.enabled and
                        self.performance_status == "严重"):
                    self.speech_manager.speak_direct(f"系统性能{self.performance_status}，建议立即检查")

            self.last_performance_report = current_time

    def reset_session(self):
        """重置会话统计"""
        self.session_start_time = time.time()
        self.performance_events = []
        self.performance_anomalies = []
        self.performance_snapshots = []
        self.gesture_counts = {}
        self.gesture_confidence_sum = {}
        self.gesture_confidence_count = {}
        self.gesture_recognition_latency = {}
        self.gesture_transitions = {}
        self.last_gesture = None
        self.error_count = 0
        self.warning_count = 0
        self.critical_count = 0
        self.drone_commands = 0
        self.successful_commands = 0
        self.failed_commands = 0
        self.command_latency_history.clear()
        self.performance_status = "优秀"
        self.performance_score = 100
        self.frame_time_history.clear()
        self.recording_sessions = 0
        self.total_trajectory_points = 0
        self.trajectory_recording_time = 0
        self.recognition_latency_distribution = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0,
            'bad': 0
        }
        self.fps_jitter_history.clear()
        self.fps_stability_score = 100
        self.system_response_times.clear()
        self.avg_system_response_time = 0

        print("✓ 性能统计会话已重置")

    def get_stats_summary(self):
        """获取统计摘要"""
        summary = {
            'fps': self.get_current_fps(),
            'avg_fps': self.get_average_fps(),
            'min_fps': min(self.fps_history) if self.fps_history else 0,
            'recognition_time': self.avg_recognition_time,
            'cpu_usage': self.get_current_cpu_usage(),
            'memory_usage': self.get_current_memory_usage(),
            'performance_status': self.performance_status,
            'performance_score': self.performance_score,
            'gesture_count': sum(self.gesture_counts.values()),
            'unique_gestures': len(self.gesture_counts),
            'fps_trend': self.get_fps_trend(),
            'cpu_trend': self.get_cpu_trend(),
            'memory_trend': self.get_memory_trend(),
            'warning_count': self.warning_count,
            'error_count': self.error_count,
            'fps_stability': self.fps_stability_score,
            'system_response_time': self.avg_system_response_time
        }

        # 添加GPU信息（如果可用）
        if self.gpu_available:
            summary['gpu_usage'] = self.get_gpu_usage()

        return summary

    def get_detailed_stats(self):
        """获取详细统计"""
        return {
            'frame_time_stats': {
                'avg': statistics.mean(self.frame_time_history) if self.frame_time_history else 0,
                'min': min(self.frame_time_history) if self.frame_time_history else 0,
                'max': max(self.frame_time_history) if self.frame_time_history else 0,
                'std': statistics.stdev(self.frame_time_history) if len(self.frame_time_history) >= 2 else 0
            },
            'recognition_time_stats': {
                'avg': self.avg_recognition_time,
                'min': self.min_recognition_time if self.min_recognition_time != float('inf') else 0,
                'max': self.max_recognition_time,
                'std': self.recognition_time_std
            },
            'resource_usage': {
                'cpu_avg': statistics.mean(self.cpu_usage_history) if self.cpu_usage_history else 0,
                'memory_avg': statistics.mean(self.memory_usage_history) if self.memory_usage_history else 0,
                'cpu_max': max(self.cpu_usage_history) if self.cpu_usage_history else 0,
                'memory_max': max(self.memory_usage_history) if self.memory_usage_history else 0
            },
            'anomaly_count': len(self.performance_anomalies),
            'snapshot_count': len(self.performance_snapshots),
            'fps_stability_stats': {
                'avg_jitter': statistics.mean(self.fps_jitter_history) if self.fps_jitter_history else 0,
                'max_jitter': max(self.fps_jitter_history) if self.fps_jitter_history else 0
            },
            'recognition_latency_distribution': dict(self.recognition_latency_distribution),
            'gesture_transition_count': len(self.gesture_transitions)
        }