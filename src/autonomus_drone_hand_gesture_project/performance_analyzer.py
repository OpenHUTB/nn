"""
性能分析器模块
负责监控和报告系统性能
作者: xiaoshiyuan888
"""

import time
import os
import csv
import numpy as np
from datetime import datetime
from collections import deque, Counter

class PerformanceAnalyzer:
    """性能分析器 - 监控和报告系统性能"""

    def __init__(self, speech_manager=None, psutil_lib=None, config=None):
        self.speech_manager = speech_manager
        self.psutil_lib = psutil_lib
        self.config = config
        self.start_time = time.time()
        self.session_start_time = time.time()

        # 帧率统计
        self.frame_times = deque(maxlen=300)
        self.frame_count = 0
        self.fps_history = deque(maxlen=100)

        # 手势识别性能
        self.gesture_recognition_times = deque(maxlen=100)
        self.avg_recognition_time = 0
        self.max_recognition_time = 0

        # 系统资源监控
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)

        # 性能事件记录
        self.performance_events = []
        self.performance_snapshots = []

        # 手势统计
        self.gesture_counts = {}
        self.gesture_confidence_sum = {}
        self.gesture_confidence_count = {}

        # 错误统计
        self.error_count = 0
        self.warning_count = 0

        # 无人机控制统计
        self.drone_commands = 0
        self.successful_commands = 0
        self.failed_commands = 0

        # 轨迹记录统计
        self.recording_sessions = 0
        self.total_trajectory_points = 0

        # 性能日志
        self.performance_log = []
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'performance_log.csv')

        # 性能阈值
        self.performance_thresholds = {
            'fps_warning': 15,
            'fps_critical': 5,
            'cpu_warning': 80,
            'cpu_critical': 90,
            'memory_warning': 85,
            'memory_critical': 95,
            'recognition_warning': 50,
            'recognition_critical': 100
        }

        # 性能状态
        self.performance_status = "良好"
        self.last_performance_report = 0
        self.auto_report_interval = 60

        print("✓ 性能分析器已初始化")

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

    def update_gesture_recognition_time(self, recognition_time_ms):
        """更新手势识别时间"""
        self.gesture_recognition_times.append(recognition_time_ms)

        # 更新平均识别时间
        if len(self.gesture_recognition_times) > 0:
            self.avg_recognition_time = np.mean(list(self.gesture_recognition_times))
            self.max_recognition_time = max(self.max_recognition_time, recognition_time_ms)

    def update_system_resources(self):
        """更新系统资源使用情况"""
        try:
            if self.psutil_lib:
                cpu_percent = self.psutil_lib.cpu_percent(interval=0.1)
                memory_percent = self.psutil_lib.virtual_memory().percent

                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory_percent)

                # 检查性能问题
                self.check_performance_issues(cpu_percent, memory_percent)
        except:
            pass

    def check_performance_issues(self, cpu_percent, memory_percent):
        """检查性能问题"""
        issues = []

        # 检查FPS
        if len(self.fps_history) > 0:
            avg_fps = np.mean(list(self.fps_history[-10:])) if len(self.fps_history) >= 10 else self.fps_history[-1]

            if avg_fps < self.performance_thresholds['fps_critical']:
                issues.append(("严重", f"帧率过低: {avg_fps:.1f} FPS"))
                self.performance_status = "严重"
            elif avg_fps < self.performance_thresholds['fps_warning']:
                issues.append(("警告", f"帧率较低: {avg_fps:.1f} FPS"))
                if self.performance_status == "良好":
                    self.performance_status = "警告"

        # 检查CPU使用率
        if cpu_percent > self.performance_thresholds['cpu_critical']:
            issues.append(("严重", f"CPU使用率过高: {cpu_percent:.1f}%"))
            self.performance_status = "严重"
        elif cpu_percent > self.performance_thresholds['cpu_warning']:
            issues.append(("警告", f"CPU使用率较高: {cpu_percent:.1f}%"))
            if self.performance_status == "良好":
                self.performance_status = "警告"

        # 检查内存使用率
        if memory_percent > self.performance_thresholds['memory_critical']:
            issues.append(("严重", f"内存使用率过高: {memory_percent:.1f}%"))
            self.performance_status = "严重"
        elif memory_percent > self.performance_thresholds['memory_warning']:
            issues.append(("警告", f"内存使用率较高: {memory_percent:.1f}%"))
            if self.performance_status == "良好":
                self.performance_status = "警告"

        # 检查手势识别时间
        if self.avg_recognition_time > self.performance_thresholds['recognition_critical']:
            issues.append(("严重", f"手势识别时间过长: {self.avg_recognition_time:.1f}ms"))
            self.performance_status = "严重"
        elif self.avg_recognition_time > self.performance_thresholds['recognition_warning']:
            issues.append(("警告", f"手势识别时间较长: {self.avg_recognition_time:.1f}ms"))
            if self.performance_status == "良好":
                self.performance_status = "警告"

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
            'session_time': time.time() - self.session_start_time
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
            'message': event['message']
        }
        self.performance_log.append(log_entry)

    def record_gesture(self, gesture, confidence):
        """记录手势统计"""
        if gesture not in self.gesture_counts:
            self.gesture_counts[gesture] = 0
            self.gesture_confidence_sum[gesture] = 0
            self.gesture_confidence_count[gesture] = 0

        self.gesture_counts[gesture] += 1
        self.gesture_confidence_sum[gesture] += confidence
        self.gesture_confidence_count[gesture] += 1

    def record_drone_command(self, success=True):
        """记录无人机命令"""
        self.drone_commands += 1
        if success:
            self.successful_commands += 1
        else:
            self.failed_commands += 1

    def record_recording_session(self, points_count=0):
        """记录录制会话"""
        self.recording_sessions += 1
        self.total_trajectory_points += points_count

    def take_snapshot(self, label=""):
        """拍摄性能快照"""
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'fps': self.get_current_fps(),
            'avg_fps': self.get_average_fps(),
            'avg_recognition_time': self.avg_recognition_time,
            'max_recognition_time': self.max_recognition_time,
            'cpu_usage': self.get_current_cpu_usage(),
            'memory_usage': self.get_current_memory_usage(),
            'gesture_counts': dict(self.gesture_counts),
            'performance_status': self.performance_status,
            'frame_count': self.frame_count,
            'session_duration': time.time() - self.session_start_time
        }
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
            '平均手势识别时间': f"{self.avg_recognition_time:.1f}ms",
            '最大手势识别时间': f"{self.max_recognition_time:.1f}ms",
            '当前CPU使用率': f"{self.get_current_cpu_usage():.1f}%",
            '当前内存使用率': f"{self.get_current_memory_usage():.1f}%",
            '性能状态': self.performance_status,
            '警告数量': self.warning_count,
            '错误数量': self.error_count,
            '无人机命令': {
                '总数': self.drone_commands,
                '成功': self.successful_commands,
                '失败': self.failed_commands,
                '成功率': f"{(self.successful_commands / self.drone_commands * 100 if self.drone_commands > 0 else 0):.1f}%"
            },
            '录制统计': {
                '会话数': self.recording_sessions,
                '总轨迹点数': self.total_trajectory_points
            }
        }

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

                gesture_stats[gesture] = {
                    '次数': count,
                    '占比': f"{(count / self.frame_count * 100 if self.frame_count > 0 else 0):.1f}%",
                    '平均置信度': f"{avg_confidence:.1%}"
                }

            report['手势统计'] = gesture_stats

            # 性能事件
            if self.performance_events:
                recent_events = list(self.performance_events)[-10:]
                report['最近性能事件'] = [
                    {
                        '时间': datetime.fromtimestamp(e['timestamp']).strftime('%H:%M:%S'),
                        '级别': e['level'],
                        '消息': e['message']
                    }
                    for e in recent_events
                ]

            # 性能快照
            if self.performance_snapshots:
                report['性能快照数'] = len(self.performance_snapshots)

            # 系统建议
            suggestions = self.generate_suggestions()
            if suggestions:
                report['优化建议'] = suggestions

        return report

    def generate_suggestions(self):
        """生成优化建议"""
        suggestions = []

        # 检查FPS
        avg_fps = self.get_average_fps()
        if avg_fps < self.performance_thresholds['fps_warning']:
            suggestions.append(f"帧率较低({avg_fps:.1f}FPS)，建议切换到'最快'性能模式")

        # 检查CPU
        cpu_usage = self.get_current_cpu_usage()
        if cpu_usage > self.performance_thresholds['cpu_warning']:
            suggestions.append(f"CPU使用率较高({cpu_usage:.1f}%)，请关闭其他占用CPU的程序")

        # 检查内存
        memory_usage = self.get_current_memory_usage()
        if memory_usage > self.performance_thresholds['memory_warning']:
            suggestions.append(f"内存使用率较高({memory_usage:.1f}%)，请关闭不必要的程序")

        # 检查识别时间
        if self.avg_recognition_time > self.performance_thresholds['recognition_warning']:
            suggestions.append(f"手势识别时间较长({self.avg_recognition_time:.1f}ms)，建议调整摄像头位置或光线")

        return suggestions

    def print_report(self, detailed=True):
        """打印性能报告"""
        report = self.generate_report(detailed)

        print("\n" + "=" * 80)
        print("📊 性能分析报告")
        print("=" * 80)

        # 基础信息
        print(f"生成时间: {report['生成时间']}")
        print(f"会话时长: {report['会话时长']}")
        print(f"总帧数: {report['总帧数']}")
        print(f"平均FPS: {report['平均FPS']}")
        print(f"当前FPS: {report['当前FPS']}")
        print(f"平均手势识别时间: {report['平均手势识别时间']}")
        print(f"最大手势识别时间: {report['最大手势识别时间']}")
        print(f"当前CPU使用率: {report['当前CPU使用率']}")
        print(f"当前内存使用率: {report['当前内存使用率']}")
        print(f"性能状态: {report['性能状态']}")

        # 无人机命令统计
        cmd_stats = report['无人机命令']
        print(f"\n无人机命令统计:")
        print(f"  总数: {cmd_stats['总数']}")
        print(f"  成功: {cmd_stats['成功']}")
        print(f"  失败: {cmd_stats['失败']}")
        print(f"  成功率: {cmd_stats['成功率']}")

        # 录制统计
        rec_stats = report['录制统计']
        print(f"\n录制统计:")
        print(f"  会话数: {rec_stats['会话数']}")
        print(f"  总轨迹点数: {rec_stats['总轨迹点数']}")

        # 详细报告
        if detailed and '手势统计' in report:
            print(f"\n手势统计:")
            for gesture, stats in report['手势统计'].items():
                print(f"  {gesture}: {stats['次数']}次 ({stats['占比']}), 平均置信度: {stats['平均置信度']}")

        # 性能事件
        if detailed and '最近性能事件' in report and report['最近性能事件']:
            print(f"\n最近性能事件:")
            for event in report['最近性能事件']:
                print(f"  [{event['时间']}] {event['级别']}: {event['消息']}")

        # 优化建议
        if detailed and '优化建议' in report and report['优化建议']:
            print(f"\n优化建议:")
            for i, suggestion in enumerate(report['优化建议'], 1):
                print(f"  {i}. {suggestion}")

        print("=" * 80)

        # 语音播报摘要
        if self.speech_manager and self.speech_manager.enabled:
            summary = (f"性能报告: 平均帧率{report['平均FPS']}，识别时间{report['平均手势识别时间']}，"
                       f"性能状态{report['性能状态']}，无人机命令成功率{cmd_stats['成功率']}")
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

    def auto_report(self):
        """自动性能报告（定期执行）"""
        current_time = time.time()
        if current_time - self.last_performance_report > self.auto_report_interval:
            # 生成简要报告
            report = self.generate_report(detailed=False)

            # 检查是否需要报告
            if (self.performance_status == "严重" or
                    self.warning_count > 5 or
                    self.error_count > 0):

                print(f"⚠ 自动性能检查: {report['性能状态']}, FPS: {report['当前FPS']}, "
                      f"CPU: {report['当前CPU使用率']}, 内存: {report['当前内存使用率']}")

                # 语音提示
                if (self.speech_manager and
                        self.speech_manager.enabled and
                        self.performance_status == "严重"):
                    self.speech_manager.speak_direct(f"系统性能{self.performance_status}，建议检查")

            self.last_performance_report = current_time

    def reset_session(self):
        """重置会话统计"""
        self.session_start_time = time.time()
        self.performance_events = []
        self.performance_snapshots = []
        self.gesture_counts = {}
        self.gesture_confidence_sum = {}
        self.gesture_confidence_count = {}
        self.error_count = 0
        self.warning_count = 0
        self.drone_commands = 0
        self.successful_commands = 0
        self.failed_commands = 0
        self.performance_status = "良好"

        print("✓ 性能统计会话已重置")

    def get_stats_summary(self):
        """获取统计摘要"""
        return {
            'fps': self.get_current_fps(),
            'avg_fps': self.get_average_fps(),
            'recognition_time': self.avg_recognition_time,
            'cpu_usage': self.get_current_cpu_usage(),
            'memory_usage': self.get_current_memory_usage(),
            'performance_status': self.performance_status,
            'gesture_count': sum(self.gesture_counts.values()),
            'unique_gestures': len(self.gesture_counts)
        }