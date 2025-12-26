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
            'high_risk_distance': 5.0,  # 高风险距离 (米)
            'medium_risk_distance': 10.0,  # 中风险距离 (米)
            'low_risk_distance': 20.0,  # 低风险距离 (米)
            'safe_speed_limit': 30.0,  # 安全速度限制 (km/h)
            'reaction_time': 1.5,  # 反应时间 (秒)
            'braking_deceleration': 6.0  # 制动减速度 (m/s²)
        }

        # 统计数据
        self.stats = {
            'total_interactions': 0,
            'high_risk_cases': 0,
            'medium_risk_cases': 0,
            'low_risk_cases': 0,
            'safe_cases': 0,
            'near_misses': 0,
            'safety_warnings': 0,
            'average_distance': 0,
            'min_distance': float('inf'),
            'max_distance': 0,
            'interaction_times': []
        }

        # 详细记录
        self.interaction_records = []
        self.warning_logs = []

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

                # 计算距离
                distance = vehicle_location.distance(pedestrian_location)

                # 计算相对速度
                pedestrian_velocity = pedestrian.get_velocity()
                relative_speed = self._calculate_relative_speed(vehicle_velocity, pedestrian_velocity)

                # 计算碰撞时间
                time_to_collision = self._calculate_ttc(distance, relative_speed)

                # 评估风险
                risk_level = self._assess_risk(distance, vehicle_speed, time_to_collision)

                interaction = {
                    'timestamp': time.time(),
                    'vehicle_id': vehicle.id,
                    'pedestrian_id': pedestrian.id,
                    'distance': distance,
                    'vehicle_speed': vehicle_speed * 3.6,  # 转换为km/h
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
                    }
                }

                current_interactions.append(interaction)

                # 更新统计
                self._update_stats(interaction)

                # 记录高风险情况
                if risk_level == 'high':
                    self._log_high_risk(interaction)

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
        if relative_speed > 0.1:  # 避免除以零
            return distance / relative_speed
        return float('inf')

    def _assess_risk(self, distance: float, speed: float, ttc: Optional[float]) -> str:
        """评估风险等级"""
        speed_kmh = speed * 3.6

        # 基于距离的风险评估
        if distance < self.safety_thresholds['high_risk_distance']:
            if ttc is not None and ttc < 2.0:
                return 'high'
            else:
                return 'medium'
        elif distance < self.safety_thresholds['medium_risk_distance']:
            if speed_kmh > self.safety_thresholds['safe_speed_limit']:
                return 'medium'
            else:
                return 'low'
        elif distance < self.safety_thresholds['low_risk_distance']:
            return 'low'
        else:
            return 'safe'

    def _update_stats(self, interaction: Dict):
        """更新统计数据"""
        self.stats['total_interactions'] += 1
        distance = interaction['distance']

        # 更新距离统计
        self.stats['average_distance'] = (
                (self.stats['average_distance'] * (self.stats['total_interactions'] - 1) + distance) /
                self.stats['total_interactions']
        )
        self.stats['min_distance'] = min(self.stats['min_distance'], distance)
        self.stats['max_distance'] = max(self.stats['max_distance'], distance)

        # 更新风险统计
        risk_level = interaction['risk_level']
        if risk_level == 'high':
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

    def _log_high_risk(self, interaction: Dict):
        """记录高风险情况"""
        warning = {
            'timestamp': datetime.now().isoformat(),
            'interaction': interaction,
            'safety_measures': self._suggest_safety_measures(interaction)
        }
        self.warning_logs.append(warning)

        # 保存高风险警告
        if len(self.warning_logs) % 10 == 0:
            self._save_warning_logs()

    def _suggest_safety_measures(self, interaction: Dict) -> List[str]:
        """建议安全措施"""
        measures = []

        if interaction['risk_level'] == 'high':
            measures.extend([
                "立即制动",
                "鸣喇叭警告",
                "准备紧急避让",
                "向其他车辆发送警告"
            ])
        elif interaction['risk_level'] == 'medium':
            measures.extend([
                "减速行驶",
                "保持警惕",
                "准备制动",
                "观察行人动向"
            ])
        elif interaction['risk_level'] == 'low':
            measures.extend([
                "保持安全距离",
                "观察周围环境",
                "准备应对突发情况"
            ])

        return measures

    def _save_interaction_report(self, interactions: List[Dict]):
        """保存交互报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.safety_dir, f"interactions_{timestamp}.json")

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_interactions': len(interactions),
            'interactions': interactions,
            'summary': {
                'high_risk': len([i for i in interactions if i['risk_level'] == 'high']),
                'medium_risk': len([i for i in interactions if i['risk_level'] == 'medium']),
                'low_risk': len([i for i in interactions if i['risk_level'] == 'low']),
                'safe': len([i for i in interactions if i['risk_level'] == 'safe'])
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

    def _generate_safety_report(self) -> Dict:
        """生成安全报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats.copy(),
            'safety_thresholds': self.safety_thresholds,
            'risk_distribution': {
                'high': self.stats['high_risk_cases'],
                'medium': self.stats['medium_risk_cases'],
                'low': self.stats['low_risk_cases'],
                'safe': self.stats['safe_cases']
            },
            'safety_score': self._calculate_safety_score(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _calculate_safety_score(self) -> float:
        """计算安全评分"""
        if self.stats['total_interactions'] == 0:
            return 100.0

        high_risk_ratio = self.stats['high_risk_cases'] / self.stats['total_interactions']
        medium_risk_ratio = self.stats['medium_risk_cases'] / self.stats['total_interactions']

        # 评分公式：基础分减去风险比例
        score = 100 - (high_risk_ratio * 60 + medium_risk_ratio * 30) * 100

        # 考虑平均距离
        if self.stats['average_distance'] > 15.0:
            score += 10
        elif self.stats['average_distance'] < 5.0:
            score -= 20

        return max(0, min(100, score))

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if self.stats['high_risk_cases'] > 0:
            recommendations.extend([
                "增加行人安全距离阈值",
                "加强车辆行人检测系统",
                "实施更严格的限速措施",
                "增加行人警告系统"
            ])

        if self.stats['average_distance'] < 10.0:
            recommendations.append("增加车辆与行人的平均距离")

        if self.stats['near_misses'] > 5:
            recommendations.append("实施紧急制动系统")

        return recommendations

    def generate_final_report(self) -> Dict:
        """生成最终报告"""
        final_report = self._generate_safety_report()

        # 添加历史数据
        final_report['historical_data'] = {
            'total_interaction_records': len(self.interaction_records),
            'total_warning_logs': len(self.warning_logs),
            'analysis_period': self._get_analysis_period()
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

        # 保存警告日志
        self._save_warning_logs()

        # 生成并保存最终报告
        self.generate_final_report()

        print(f"行人安全数据已保存到: {self.safety_dir}")