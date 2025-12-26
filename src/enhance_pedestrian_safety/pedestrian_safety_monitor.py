import json
import os
import time
import math
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import carla


class PedestrianSafetyMonitor:
    """行人安全监控器"""

    def __init__(self, world, output_dir):
        self.world = world
        self.output_dir = output_dir
        self.safety_dir = os.path.join(output_dir, "safety_reports")
        os.makedirs(self.safety_dir, exist_ok=True)

        # 安全参数
        self.safety_thresholds = {
            'critical_distance': 2.0,  # 临界距离 (米)
            'high_risk_distance': 5.0,  # 高风险距离 (米)
            'medium_risk_distance': 10.0,  # 中风险距离 (米)
            'low_risk_distance': 20.0,  # 低风险距离 (米)
            'safe_speed_limit': 30.0,  # 安全速度限制 (km/h)
            'reaction_time': 1.5,  # 反应时间 (秒)
            'braking_deceleration': 6.0,  # 制动减速度 (m/s²)
            'ttc_critical': 1.0,  # 临界碰撞时间 (秒)
            'ttc_high': 2.0,  # 高风险碰撞时间 (秒)
            'ttc_medium': 3.0  # 中风险碰撞时间 (秒)
        }

        # 统计数据
        self.stats = {
            'total_interactions': 0,
            'critical_cases': 0,
            'high_risk_cases': 0,
            'medium_risk_cases': 0,
            'low_risk_cases': 0,
            'safe_cases': 0,
            'near_misses': 0,
            'safety_warnings': 0,
            'average_distance': 0,
            'min_distance': float('inf'),
            'max_distance': 0,
            'average_ttc': 0,
            'min_ttc': float('inf'),
            'interaction_times': [],
            'vehicle_speeds': [],
            'pedestrian_speeds': []
        }

        # 详细记录
        self.interaction_records = []
        self.warning_logs = []
        self.critical_events = []

    def check_pedestrian_safety(self) -> Dict:
        """检查行人安全"""
        vehicles = self._get_vehicles()
        pedestrians = self._get_pedestrians()

        current_interactions = []

        for vehicle in vehicles:
            vehicle_location = vehicle.get_location()
            vehicle_velocity = vehicle.get_velocity()
            vehicle_speed = math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)

            for pedestrian in pedestrians:
                pedestrian_location = pedestrian.get_location()
                pedestrian_velocity = pedestrian.get_velocity()

                # 计算距离
                distance = vehicle_location.distance(pedestrian_location)

                # 计算相对速度
                relative_speed = self._calculate_relative_speed(vehicle_velocity, pedestrian_velocity)

                # 计算碰撞时间
                time_to_collision = self._calculate_ttc(distance, relative_speed)

                # 评估风险
                risk_level = self._assess_risk(distance, vehicle_speed, time_to_collision, relative_speed)

                interaction = {
                    'timestamp': time.time(),
                    'vehicle_id': vehicle.id,
                    'pedestrian_id': pedestrian.id,
                    'distance': distance,
                    'vehicle_speed': vehicle_speed * 3.6,  # 转换为km/h
                    'pedestrian_speed': math.sqrt(pedestrian_velocity.x ** 2 + pedestrian_velocity.y ** 2) * 3.6,
                    'relative_speed': relative_speed * 3.6,
                    'time_to_collision': time_to_collision if time_to_collision < 100 else None,
                    'risk_level': risk_level,
                    'vehicle_location': {
                        'x': vehicle_location.x,
                        'y': vehicle_location.y,
                        'z': vehicle_location.z
                    },
                    'pedestrian_location': {
                        'x': pedestrian_location.x,
                        'y': pedestrian_location.y,
                        'z': pedestrian_location.z
                    },
                    'safety_measures': self._suggest_safety_measures(distance, vehicle_speed, time_to_collision,
                                                                     risk_level)
                }

                current_interactions.append(interaction)

                # 更新统计
                self._update_stats(interaction)

                # 记录高风险和临界情况
                if risk_level in ['critical', 'high']:
                    self._log_risk_event(interaction)

        # 保存当前检查结果
        if current_interactions:
            self._save_interaction_report(current_interactions)

        return self._generate_safety_report()

    def _get_vehicles(self) -> List[carla.Actor]:
        """获取所有车辆"""
        return [actor for actor in self.world.get_actors() if 'vehicle' in actor.type_id]

    def _get_pedestrians(self) -> List[carla.Actor]:
        """获取所有行人"""
        return [actor for actor in self.world.get_actors() if 'walker' in actor.type_id]

    def _calculate_relative_speed(self, v1: carla.Vector3D, v2: carla.Vector3D) -> float:
        """计算相对速度"""
        return math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2 + (v1.z - v2.z) ** 2)

    def _calculate_ttc(self, distance: float, relative_speed: float) -> float:
        """计算碰撞时间 (Time to Collision)"""
        if relative_speed > 0.1:
            return distance / relative_speed
        return float('inf')

    def _assess_risk(self, distance: float, speed: float, ttc: Optional[float], relative_speed: float) -> str:
        """评估风险等级"""
        speed_kmh = speed * 3.6

        # 基于碰撞时间的风险评估
        if ttc is not None:
            if ttc < self.safety_thresholds['ttc_critical']:
                return 'critical'
            elif ttc < self.safety_thresholds['ttc_high']:
                return 'high'
            elif ttc < self.safety_thresholds['ttc_medium']:
                return 'medium'

        # 基于距离的风险评估
        if distance < self.safety_thresholds['critical_distance']:
            return 'critical'
        elif distance < self.safety_thresholds['high_risk_distance']:
            if speed_kmh > self.safety_thresholds['safe_speed_limit']:
                return 'high'
            else:
                return 'medium'
        elif distance < self.safety_thresholds['medium_risk_distance']:
            if speed_kmh > self.safety_thresholds['safe_speed_limit'] * 1.5:
                return 'medium'
            else:
                return 'low'
        elif distance < self.safety_thresholds['low_risk_distance']:
            return 'low'
        else:
            return 'safe'

    def _suggest_safety_measures(self, distance: float, speed: float, ttc: Optional[float], risk_level: str) -> List[
        str]:
        """建议安全措施"""
        measures = []
        speed_kmh = speed * 3.6

        if risk_level == 'critical':
            measures.extend([
                "立即紧急制动",
                "鸣喇叭警告行人",
                "准备紧急避让",
                "向其他车辆发送紧急警告",
                "记录事故数据"
            ])
        elif risk_level == 'high':
            measures.extend([
                "立即减速至20km/h以下",
                "保持警惕，准备制动",
                "观察行人动向",
                "准备避让",
                "向附近车辆发送警告"
            ])
        elif risk_level == 'medium':
            measures.extend([
                "减速至安全速度",
                "保持安全距离",
                "观察周围环境",
                "准备应对突发情况",
                "评估避让路径"
            ])
        elif risk_level == 'low':
            measures.extend([
                "保持当前速度",
                "注意观察",
                "准备减速",
                "保持安全车距"
            ])
        else:
            measures.append("正常行驶，保持警惕")

        # 添加基于距离的额外建议
        if distance < 3.0:
            measures.append("保持极端警惕，准备紧急操作")
        elif distance < 5.0:
            measures.append("准备随时制动")

        # 添加基于车速的额外建议
        if speed_kmh > 50.0:
            measures.append("车速过快，建议减速")
        elif speed_kmh > 30.0:
            measures.append("注意控制车速")

        return measures

    def _update_stats(self, interaction: Dict):
        """更新统计数据"""
        self.stats['total_interactions'] += 1
        distance = interaction['distance']
        ttc = interaction.get('time_to_collision')

        # 更新距离统计
        self.stats['average_distance'] = (
                (self.stats['average_distance'] * (self.stats['total_interactions'] - 1) + distance) /
                self.stats['total_interactions']
        )
        self.stats['min_distance'] = min(self.stats['min_distance'], distance)
        self.stats['max_distance'] = max(self.stats['max_distance'], distance)

        # 更新碰撞时间统计
        if ttc is not None and ttc < 100:
            self.stats['average_ttc'] = (
                    (self.stats['average_ttc'] * (self.stats['total_interactions'] - 1) + ttc) /
                    self.stats['total_interactions']
            )
            self.stats['min_ttc'] = min(self.stats['min_ttc'], ttc)

        # 更新速度统计
        self.stats['vehicle_speeds'].append(interaction['vehicle_speed'])
        self.stats['pedestrian_speeds'].append(interaction['pedestrian_speed'])

        # 更新风险统计
        risk_level = interaction['risk_level']
        if risk_level == 'critical':
            self.stats['critical_cases'] += 1
            self.stats['near_misses'] += 1
            self.stats['safety_warnings'] += 1
        elif risk_level == 'high':
            self.stats['high_risk_cases'] += 1
            self.stats['near_misses'] += 1
            self.stats['safety_warnings'] += 1
        elif risk_level == 'medium':
            self.stats['medium_risk_cases'] += 1
            self.stats['safety_warnings'] += 1
        elif risk_level == 'low':
            self.stats['low_risk_cases'] += 1
        else:
            self.stats['safe_cases'] += 1

        # 记录交互时间
        self.stats['interaction_times'].append(interaction['timestamp'])

        # 添加到详细记录
        self.interaction_records.append(interaction)

        # 限制记录数量
        if len(self.interaction_records) > 1000:
            self.interaction_records = self.interaction_records[-1000:]

    def _log_risk_event(self, interaction: Dict):
        """记录风险事件"""
        event_type = 'critical' if interaction['risk_level'] == 'critical' else 'high_risk'

        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'interaction': interaction,
            'safety_measures': interaction.get('safety_measures', []),
            'environment': {
                'weather': str(self.world.get_weather()),
                'time_of_day': self._get_time_of_day()
            }
        }

        if event_type == 'critical':
            self.critical_events.append(event)
            # 保存临界事件
            if len(self.critical_events) % 5 == 0:
                self._save_critical_events()
        else:
            self.warning_logs.append(event)

        # 保存警告日志
        if len(self.warning_logs) % 10 == 0:
            self._save_warning_logs()

    def _get_time_of_day(self) -> str:
        """获取当前时间"""
        weather = self.world.get_weather()
        sun_altitude = weather.sun_altitude_angle

        if sun_altitude > 45:
            return 'day'
        elif sun_altitude > 0:
            return 'sunset'
        else:
            return 'night'

    def _save_interaction_report(self, interactions: List[Dict]):
        """保存交互报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.safety_dir, f"interactions_{timestamp}.json")

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_interactions': len(interactions),
            'interactions': interactions,
            'summary': {
                'critical': len([i for i in interactions if i['risk_level'] == 'critical']),
                'high_risk': len([i for i in interactions if i['risk_level'] == 'high']),
                'medium_risk': len([i for i in interactions if i['risk_level'] == 'medium']),
                'low_risk': len([i for i in interactions if i['risk_level'] == 'low']),
                'safe': len([i for i in interactions if i['risk_level'] == 'safe'])
            },
            'environment': {
                'weather': str(self.world.get_weather()),
                'time_of_day': self._get_time_of_day()
            }
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _save_warning_logs(self):
        """保存警告日志"""
        if not self.warning_logs:
            return

        warning_file = os.path.join(self.safety_dir, "warning_logs.json")
        with open(warning_file, 'w', encoding='utf-8') as f:
            json.dump(self.warning_logs, f, indent=2, ensure_ascii=False)

    def _save_critical_events(self):
        """保存临界事件"""
        if not self.critical_events:
            return

        critical_file = os.path.join(self.safety_dir, "critical_events.json")
        with open(critical_file, 'w', encoding='utf-8') as f:
            json.dump(self.critical_events, f, indent=2, ensure_ascii=False)

    def _generate_safety_report(self) -> Dict:
        """生成安全报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats.copy(),
            'safety_thresholds': self.safety_thresholds,
            'risk_distribution': {
                'critical': self.stats['critical_cases'],
                'high': self.stats['high_risk_cases'],
                'medium': self.stats['medium_risk_cases'],
                'low': self.stats['low_risk_cases'],
                'safe': self.stats['safe_cases']
            },
            'safety_score': self._calculate_safety_score(),
            'recommendations': self._generate_recommendations(),
            'performance_metrics': {
                'average_vehicle_speed': np.mean(self.stats['vehicle_speeds']) if self.stats['vehicle_speeds'] else 0,
                'average_pedestrian_speed': np.mean(self.stats['pedestrian_speeds']) if self.stats[
                    'pedestrian_speeds'] else 0,
                'max_vehicle_speed': max(self.stats['vehicle_speeds']) if self.stats['vehicle_speeds'] else 0,
                'interaction_frequency': len(self.stats['interaction_times']) / max(1, (
                    time.time() - min(self.stats['interaction_times']) if self.stats['interaction_times'] else 1))
            }
        }

        return report

    def _calculate_safety_score(self) -> float:
        """计算安全评分"""
        if self.stats['total_interactions'] == 0:
            return 100.0

        critical_ratio = self.stats['critical_cases'] / self.stats['total_interactions']
        high_risk_ratio = self.stats['high_risk_cases'] / self.stats['total_interactions']
        medium_risk_ratio = self.stats['medium_risk_cases'] / self.stats['total_interactions']

        # 评分公式：基础分减去风险比例加权
        score = 100 - (critical_ratio * 80 + high_risk_ratio * 50 + medium_risk_ratio * 20) * 100

        # 考虑平均距离
        if self.stats['average_distance'] > 15.0:
            score += 10
        elif self.stats['average_distance'] < 5.0:
            score -= 20
        elif self.stats['average_distance'] < 2.0:
            score -= 40

        # 考虑平均车速
        if self.stats['vehicle_speeds']:
            avg_speed = np.mean(self.stats['vehicle_speeds'])
            if avg_speed > 50.0:
                score -= 15
            elif avg_speed > 30.0:
                score -= 5
            elif avg_speed < 20.0:
                score += 10

        return max(0, min(100, score))

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if self.stats['critical_cases'] > 0:
            recommendations.extend([
                "立即改进行人检测系统",
                "增加紧急制动系统",
                "实施更严格的限速措施",
                "增加行人预警系统",
                "记录和分析所有临界事件"
            ])

        if self.stats['high_risk_cases'] > 5:
            recommendations.extend([
                "增加行人安全距离阈值",
                "加强车辆行人检测系统",
                "实施自动减速系统",
                "增加行人警告频率"
            ])

        if self.stats['average_distance'] < 10.0:
            recommendations.append("增加车辆与行人的平均距离")

        if self.stats['near_misses'] > 3:
            recommendations.append("实施紧急避让系统")

        if self.stats['vehicle_speeds']:
            avg_speed = np.mean(self.stats['vehicle_speeds'])
            if avg_speed > 40.0:
                recommendations.append("降低平均车速，特别是在行人密集区域")

        # 基于碰撞时间的建议
        if self.stats['min_ttc'] < 2.0:
            recommendations.append("改进碰撞时间预测算法")

        return recommendations

    def generate_final_report(self) -> Dict:
        """生成最终报告"""
        final_report = self._generate_safety_report()

        # 添加历史数据
        final_report['historical_data'] = {
            'total_interaction_records': len(self.interaction_records),
            'total_warning_logs': len(self.warning_logs),
            'total_critical_events': len(self.critical_events),
            'analysis_period': self._get_analysis_period()
        }

        # 添加详细分析
        final_report['detailed_analysis'] = {
            'distance_analysis': {
                'average': self.stats['average_distance'],
                'min': self.stats['min_distance'],
                'max': self.stats['max_distance'],
                'std': np.std([r['distance'] for r in self.interaction_records]) if self.interaction_records else 0
            },
            'speed_analysis': {
                'vehicle_avg': np.mean(self.stats['vehicle_speeds']) if self.stats['vehicle_speeds'] else 0,
                'pedestrian_avg': np.mean(self.stats['pedestrian_speeds']) if self.stats['pedestrian_speeds'] else 0,
                'vehicle_std': np.std(self.stats['vehicle_speeds']) if self.stats['vehicle_speeds'] else 0
            },
            'ttc_analysis': {
                'average': self.stats['average_ttc'],
                'min': self.stats['min_ttc'],
                'percentage_below_2s': len(
                    [r for r in self.interaction_records if r.get('time_to_collision', 100) < 2.0]) / max(1,
                                                                                                          len(self.interaction_records)) * 100
            }
        }

        # 保存最终报告
        final_file = os.path.join(self.safety_dir, "final_safety_report.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        return final_report

    def _get_analysis_period(self) -> Dict:
        """获取分析时间段"""
        if not self.stats['interaction_times']:
            return {'start': None, 'end': None, 'duration': 0}

        start_time = min(self.stats['interaction_times'])
        end_time = max(self.stats['interaction_times'])
        duration = end_time - start_time

        return {
            'start': datetime.fromtimestamp(start_time).isoformat(),
            'end': datetime.fromtimestamp(end_time).isoformat(),
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'duration_hours': duration / 3600
        }

    def save_data(self):
        """保存所有数据"""
        # 保存统计数据
        stats_file = os.path.join(self.safety_dir, "safety_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        # 保存详细记录
        if self.interaction_records:
            records_file = os.path.join(self.safety_dir, "interaction_records.json")
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(self.interaction_records, f, indent=2, ensure_ascii=False)

        # 保存警告日志和临界事件
        self._save_warning_logs()
        self._save_critical_events()

        # 生成并保存最终报告
        final_report = self.generate_final_report()

        print(f"行人安全数据已保存到: {self.safety_dir}")
        print(f"安全评分: {final_report['safety_score']:.1f}/100")
        print(f"高风险事件: {final_report['risk_distribution']['high']}次")
        print(f"临界事件: {final_report['risk_distribution']['critical']}次")