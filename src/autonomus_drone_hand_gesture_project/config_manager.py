"""
配置管理器模块
负责加载、保存和管理系统配置
作者: xiaoshiyuan888
"""

import os
import json
import numpy as np
import cv2


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(current_dir, 'gesture_config.json')

        # 性能模式配置
        self.performance_modes = {
            'fast': {
                'name': '最快',
                'description': '性能优先，降低识别精度换取更高帧率',
                'detection_interval': 2,
                'smooth_frames': 3,
                'min_confidence': 0.5,
                'resize_factor': 0.5,
                'skin_detection_enabled': True,
                'background_subtraction_enabled': False,
                'contour_simplify_epsilon': 0.03,
                'history_size': 10,
                'gesture_stability_threshold': 3,
                'color': (0, 255, 0),
            },
            'balanced': {
                'name': '平衡',
                'description': '平衡性能与精度，适用于大多数场景',
                'detection_interval': 1,
                'smooth_frames': 5,
                'min_confidence': 0.6,
                'resize_factor': 0.75,
                'skin_detection_enabled': True,
                'background_subtraction_enabled': True,
                'contour_simplify_epsilon': 0.02,
                'history_size': 15,
                'gesture_stability_threshold': 5,
                'color': (255, 165, 0),
            },
            'accurate': {
                'name': '最准',
                'description': '精度优先，提供最准确的手势识别',
                'detection_interval': 1,
                'smooth_frames': 7,
                'min_confidence': 0.7,
                'resize_factor': 1.0,
                'skin_detection_enabled': True,
                'background_subtraction_enabled': True,
                'contour_simplify_epsilon': 0.01,
                'history_size': 20,
                'gesture_stability_threshold': 7,
                'color': (255, 0, 0),
            }
        }

        self.default_config = {
            'camera': {
                'index': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'gesture': {
                'skin_lower_h': 0,
                'skin_upper_h': 25,
                'skin_lower_s': 30,
                'skin_upper_s': 255,
                'skin_lower_v': 60,
                'skin_upper_v': 255,
                'min_hand_area': 2000,
                'max_hand_area': 30000,
                'hand_ratio_threshold': 1.5,
                'defect_distance_threshold': 20,
                'palm_circle_radius_ratio': 0.3,
                'transition_threshold': 0.3,
                'position_stability_weight': 0.4,
                'gesture_cooldown': 0.5,
                # 新增手势稳定性分析配置
                'stability_window_size': 15,
                'enable_gesture_stability_analysis': True,
            },
            'drone': {
                'velocity': 2.5,
                'duration': 0.3,
                'altitude': -10.0,
                'control_interval': 0.3
            },
            'display': {
                'show_fps': True,
                'show_confidence': True,
                'show_help': True,
                'show_contours': True,
                'show_bbox': True,
                'show_fingertips': True,
                'show_palm_center': True,
                'show_hand_direction': True,
                'show_debug_info': False,
                'show_speech_status': True,
                'show_gesture_history': True,
                'show_stability_indicator': True,
                'show_trajectory': True,
                'show_recording_status': True,
                'show_performance_mode': True,
                'show_performance_stats': True,
                'show_system_resources': True,
                'show_advanced_gestures': True,
                # 新增性能可视化配置
                'show_performance_visualization': True,
                'performance_visualization_mode': 'charts',  # 'none', 'charts', 'gauges'
                'show_gesture_stability_info': True,
            },
            'performance': {
                'target_fps': 30,
                'resize_factor': 1.0,
                'enable_multiprocessing': False,
                'mode': 'balanced',
                'current_mode_index': 1,
                'modes': ['fast', 'balanced', 'accurate'],
                'auto_report_interval': 60,
                'enable_performance_monitor': True,
                # 新增性能监控配置
                'enable_gpu_monitoring': True,
                'enable_performance_prediction': True,
                'enable_anomaly_detection': True,
                'performance_visualization': True,
                'detailed_performance_charts': False,
            },
            'calibration': {
                'auto_calibrate_skin': True,
                'skin_calibration_frames': 30,
                'hand_size_calibration': True,
                # 新增自动校准配置
                'auto_adjust_parameters': True,
                'calibration_interval': 300,  # 每5分钟校准一次
            },
            'speech': {
                'enabled': True,
                'volume': 1.0,
                'rate': 150,
                'announce_gestures': True,
                'announce_connections': True,
                'announce_flight_events': True,
                'announce_gesture_changes': True,
                'announce_hand_status': True,
                'announce_performance': True,
                'announce_recording_events': True,
                'announce_performance_mode': True,
                'announce_performance_events': True,
                'min_gesture_confidence': 0.7,
                'gesture_start_threshold': 3,
                'gesture_end_threshold': 10,
                # 新增语音配置
                'announce_stability_status': True,
                'announce_visualization_changes': True,
                'performance_warnings_enabled': True,
            },
            'recording': {
                'auto_save_interval': 5,
                'max_trajectory_points': 1000,
                'show_trajectory': True,
                'trajectory_thickness': 2,
                'trajectory_max_length': 100,
                'default_save_dir': 'trajectories',
                # 新增录制配置
                'auto_playback_after_recording': False,
                'save_compressed_trajectories': True,
                'max_auto_save_files': 10,
            },
            'stability': {
                # 新增稳定性分析配置
                'enable_stability_analysis': True,
                'stability_update_interval': 1.0,
                'min_stability_score': 0.6,
                'max_transition_count': 10,
                'stability_notification_enabled': True,
                'auto_reset_on_unstable': True,
            },
            'visualization': {
                # 新增可视化配置
                'chart_update_rate': 2,  # 每2帧更新一次图表
                'gauge_update_rate': 5,  # 每5帧更新一次仪表盘
                'chart_history_length': 30,
                'show_chart_grid': True,
                'chart_background_opacity': 0.7,
                'performance_colors': {
                    'excellent': (0, 255, 0),
                    'good': (100, 255, 100),
                    'fair': (255, 255, 0),
                    'poor': (255, 165, 0),
                    'critical': (255, 0, 0)
                }
            }
        }

        self.config = self.load_config()
        self.skin_calibration_data = []
        self.hand_size_calibration_done = False
        self.reference_hand_size = 0

        # 新增：可视化状态
        self.visualization_enabled = self.get('display', 'show_performance_visualization')
        self.visualization_mode = self.get('display', 'performance_visualization_mode')

    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    config = self.default_config.copy()
                    self._merge_config(config, loaded_config)
                    print("✓ 从文件加载配置")
                    return config
            except Exception as e:
                print(f"⚠ 加载配置失败: {e}, 使用默认配置")
                return self.default_config.copy()
        else:
            print("✓ 使用默认配置")
            return self.default_config.copy()

    def _merge_config(self, base, update):
        """递归合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print("✓ 配置已保存")
        except Exception as e:
            print(f"⚠ 保存配置失败: {e}")

    def get(self, *keys):
        """获取配置值"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def set(self, *keys, value):
        """设置配置值"""
        if len(keys) == 0:
            return

        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        self.save_config()

        # 更新内部状态
        if len(keys) == 2:
            if keys[0] == 'display' and keys[1] == 'show_performance_visualization':
                self.visualization_enabled = value
            elif keys[0] == 'display' and keys[1] == 'performance_visualization_mode':
                self.visualization_mode = value

    def get_performance_mode_config(self, mode=None):
        """获取性能模式配置"""
        if mode is None:
            mode = self.get('performance', 'mode')

        if mode in self.performance_modes:
            return self.performance_modes[mode]
        else:
            return self.performance_modes['balanced']

    def get_current_performance_mode(self):
        """获取当前性能模式"""
        mode = self.get('performance', 'mode')
        if mode in self.performance_modes:
            return mode
        return 'balanced'

    def set_performance_mode(self, mode):
        """设置性能模式"""
        if mode in self.performance_modes:
            self.set('performance', 'mode', value=mode)

            # 更新当前模式索引
            modes = self.get('performance', 'modes')
            if modes and mode in modes:
                index = modes.index(mode)
                self.set('performance', 'current_mode_index', value=index)

            print(f"✓ 性能模式设置为: {self.performance_modes[mode]['name']}")
            return True
        return False

    def cycle_performance_mode(self):
        """循环切换性能模式"""
        modes = self.get('performance', 'modes')
        if not modes:
            modes = ['fast', 'balanced', 'accurate']

        current_index = self.get('performance', 'current_mode_index')
        if current_index is None:
            current_index = 0

        # 计算下一个模式索引
        next_index = (current_index + 1) % len(modes)
        next_mode = modes[next_index]

        # 设置新模式
        self.set('performance', 'current_mode_index', value=next_index)
        return self.set_performance_mode(next_mode)

    def calibrate_skin_color(self, frame, hand_mask):
        """自动校准肤色范围"""
        if not self.get('calibration', 'auto_calibrate_skin'):
            return

        if len(self.skin_calibration_data) < self.get('calibration', 'skin_calibration_frames'):
            # 转换到HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 获取肤色区域的HSV值
            skin_pixels = hsv[hand_mask > 0]

            if len(skin_pixels) > 100:  # 确保有足够的像素
                self.skin_calibration_data.append(skin_pixels)

        if len(self.skin_calibration_data) == self.get('calibration', 'skin_calibration_frames'):
            # 计算肤色范围
            all_skin_pixels = np.vstack(self.skin_calibration_data)

            h_min, h_max = np.percentile(all_skin_pixels[:, 0], [2, 98])
            s_min, s_max = np.percentile(all_skin_pixels[:, 1], [2, 98])
            v_min, v_max = np.percentile(all_skin_pixels[:, 2], [2, 98])

            # 更新配置
            self.set('gesture', 'skin_lower_h', value=int(max(0, h_min - 5)))
            self.set('gesture', 'skin_upper_h', value=int(min(180, h_max + 5)))
            self.set('gesture', 'skin_lower_s', value=int(max(0, s_min - 10)))
            self.set('gesture', 'skin_upper_s', value=int(min(255, s_max + 10)))
            self.set('gesture', 'skin_lower_v', value=int(max(0, v_min - 10)))
            self.set('gesture', 'skin_upper_v', value=int(min(255, v_max + 10)))

            print("✓ 肤色校准完成")
            print(f"  肤色范围: H[{self.get('gesture', 'skin_lower_h')}-{self.get('gesture', 'skin_upper_h')}], "
                  f"S[{self.get('gesture', 'skin_lower_s')}-{self.get('gesture', 'skin_upper_s')}], "
                  f"V[{self.get('gesture', 'skin_lower_v')}-{self.get('gesture', 'skin_upper_v')}]")

    def calibrate_hand_size(self, hand_area):
        """校准手部大小"""
        if not self.get('calibration', 'hand_size_calibration') or self.hand_size_calibration_done:
            return

        if hand_area > 0:
            self.reference_hand_size = hand_area
            self.hand_size_calibration_done = True
            print(f"✓ 手部大小校准完成: {self.reference_hand_size:.0f} 像素")

    def toggle_visualization_mode(self):
        """切换可视化模式"""
        modes = ['none', 'charts', 'gauges']
        current_mode = self.get('display', 'performance_visualization_mode')

        if current_mode not in modes:
            current_mode = 'none'

        current_index = modes.index(current_mode)
        next_index = (current_index + 1) % len(modes)
        next_mode = modes[next_index]

        self.set('display', 'performance_visualization_mode', value=next_mode)
        self.visualization_mode = next_mode

        print(f"✓ 可视化模式切换为: {next_mode}")
        return next_mode

    def toggle_visualization(self):
        """切换可视化显示"""
        current = self.get('display', 'show_performance_visualization')
        new_value = not current
        self.set('display', 'show_performance_visualization', value=new_value)
        self.visualization_enabled = new_value

        print(f"✓ 性能可视化: {'开启' if new_value else '关闭'}")
        return new_value

    def get_visualization_status(self):
        """获取可视化状态"""
        return {
            'enabled': self.visualization_enabled,
            'mode': self.visualization_mode,
            'show_chart_grid': self.get('visualization', 'show_chart_grid'),
            'chart_background_opacity': self.get('visualization', 'chart_background_opacity')
        }

    def get_stability_config(self):
        """获取稳定性分析配置"""
        return {
            'enabled': self.get('stability', 'enable_stability_analysis'),
            'window_size': self.get('gesture', 'stability_window_size'),
            'update_interval': self.get('stability', 'stability_update_interval'),
            'min_score': self.get('stability', 'min_stability_score'),
            'auto_reset': self.get('stability', 'auto_reset_on_unstable')
        }

    def get_visualization_config(self):
        """获取可视化配置"""
        return {
            'chart_update_rate': self.get('visualization', 'chart_update_rate'),
            'gauge_update_rate': self.get('visualization', 'gauge_update_rate'),
            'chart_history_length': self.get('visualization', 'chart_history_length'),
            'show_grid': self.get('visualization', 'show_chart_grid'),
            'background_opacity': self.get('visualization', 'chart_background_opacity'),
            'colors': self.get('visualization', 'performance_colors')
        }

    def validate_config(self):
        """验证配置有效性"""
        issues = []

        # 检查性能模式
        current_mode = self.get('performance', 'mode')
        if current_mode not in self.performance_modes:
            issues.append(f"无效的性能模式: {current_mode}")
            self.set('performance', 'mode', value='balanced')

        # 检查可视化模式
        viz_mode = self.get('display', 'performance_visualization_mode')
        valid_viz_modes = ['none', 'charts', 'gauges']
        if viz_mode not in valid_viz_modes:
            issues.append(f"无效的可视化模式: {viz_mode}")
            self.set('display', 'performance_visualization_mode', value='charts')

        # 检查手势稳定性窗口大小
        window_size = self.get('gesture', 'stability_window_size')
        if not isinstance(window_size, int) or window_size < 5 or window_size > 50:
            issues.append(f"无效的手势稳定性窗口大小: {window_size}")
            self.set('gesture', 'stability_window_size', value=15)

        if issues:
            print("⚠ 配置验证发现问题:")
            for issue in issues:
                print(f"  - {issue}")
            print("✓ 已自动修复配置问题")

        return len(issues) == 0

    def reset_to_defaults(self, section=None):
        """重置配置为默认值"""
        if section is None:
            self.config = self.default_config.copy()
            print("✓ 所有配置已重置为默认值")
        elif section in self.default_config:
            self.config[section] = self.default_config[section].copy()
            print(f"✓ {section} 配置已重置为默认值")
        else:
            print(f"⚠ 无效的配置节: {section}")
            return False

        self.save_config()
        return True

    def export_config(self, filename=None):
        """导出配置到文件"""
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(current_dir, f'gesture_config_backup_{timestamp}.json')

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"📤 配置已导出到: {filename}")
            return True
        except Exception as e:
            print(f"❌ 导出配置失败: {e}")
            return False

    def import_config(self, filename):
        """从文件导入配置"""
        try:
            if not os.path.exists(filename):
                print(f"❌ 配置文件不存在: {filename}")
                return False

            with open(filename, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # 合并配置
            self._merge_config(self.config, loaded_config)
            self.save_config()

            print(f"📥 配置已从文件导入: {filename}")
            return True
        except Exception as e:
            print(f"❌ 导入配置失败: {e}")
            return False

# 用于测试配置管理器
if __name__ == "__main__":
    import time

    config = ConfigManager()

    print("当前配置状态:")
    print(f"  性能模式: {config.get('performance', 'mode')}")
    print(f"  可视化模式: {config.get('display', 'performance_visualization_mode')}")
    print(f"  手势稳定性分析: {config.get('gesture', 'enable_gesture_stability_analysis')}")
    print(f"  GPU监控: {config.get('performance', 'enable_gpu_monitoring')}")

    # 验证配置
    config.validate_config()