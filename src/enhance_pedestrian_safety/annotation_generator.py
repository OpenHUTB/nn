import json
import os
import numpy as np
from datetime import datetime
import threading


class AnnotationGenerator:
    """标注生成器（行人安全增强版）"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(self.annotations_dir, exist_ok=True)

        self.frame_annotations = {}
        self.object_counter = 0
        self.lock = threading.RLock()

        # 行人安全相关统计
        self.safety_stats = {
            'near_misses': 0,
            'critical_interactions': 0,
            'pedestrian_trajectories': {},
            'risk_zones': []
        }

        # 高风险区域定义
        self.high_risk_zones = []  # 学校、人行横道等区域

    def detect_objects(self, world, frame_num, timestamp):
        """检测场景中的物体（行人安全增强）"""
        annotations = {
            'frame_id': frame_num,
            'timestamp': timestamp,
            'objects': [],
            'camera_info': {},
            'safety_info': {
                'pedestrian_count': 0,
                'vehicle_count': 0,
                'high_risk_interactions': 0,
                'near_miss_count': 0,
                'pedestrian_distances': [],
                'risk_level': 'low'
            }
        }

        try:
            actors = world.get_actors()
            pedestrians = []
            vehicles = []

            for actor in actors:
                obj_type = actor.type_id

                if 'vehicle' in obj_type or 'walker' in obj_type:
                    obj_info = self._extract_object_info(actor)
                    if obj_info:
                        annotations['objects'].append(obj_info)
                        self.object_counter += 1

                        # 分类统计
                        if 'walker' in obj_type:
                            annotations['safety_info']['pedestrian_count'] += 1
                            pedestrians.append({
                                'id': actor.id,
                                'location': obj_info['location'],
                                'velocity': obj_info['velocity']
                            })
                        elif 'vehicle' in obj_type:
                            annotations['safety_info']['vehicle_count'] += 1
                            vehicles.append({
                                'id': actor.id,
                                'location': obj_info['location'],
                                'velocity': obj_info['velocity'],
                                'type': obj_info['class']
                            })

            # 检测高风险交互和近碰撞
            risk_analysis = self._analyze_pedestrian_risk(pedestrians, vehicles)
            annotations['safety_info']['high_risk_interactions'] = risk_analysis['high_risk_count']
            annotations['safety_info']['near_miss_count'] = risk_analysis['near_miss_count']
            annotations['safety_info']['pedestrian_distances'] = risk_analysis['distances']
            annotations['safety_info']['risk_level'] = risk_analysis['risk_level']

            # 跟踪行人轨迹
            self._track_pedestrian_trajectories(frame_num, pedestrians)

            # 检测高风险区域
            risk_zones = self._detect_risk_zones(pedestrians, vehicles)
            annotations['safety_info']['risk_zones'] = risk_zones

            self._save_annotations(frame_num, annotations)
            return annotations

        except Exception as e:
            print(f"物体检测失败: {e}")
            return annotations

    def _extract_object_info(self, actor):
        """提取物体信息（增强版）"""
        try:
            bbox = actor.bounding_box
            location = actor.get_location()
            velocity = actor.get_velocity()
            transform = actor.get_transform()

            # 计算速度大小
            speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6  # 转换为km/h

            obj_info = {
                'id': actor.id,
                'type': actor.type_id,
                'class': self._get_object_class(actor.type_id),
                'location': {
                    'x': float(location.x),
                    'y': float(location.y),
                    'z': float(location.z)
                },
                'velocity': {
                    'x': float(velocity.x),
                    'y': float(velocity.y),
                    'z': float(velocity.z),
                    'speed_kmh': float(speed)
                },
                'heading': float(transform.rotation.yaw),
                'bounding_box': {
                    'extent': {
                        'x': float(bbox.extent.x),
                        'y': float(bbox.extent.y),
                        'z': float(bbox.extent.z)
                    }
                },
                'attributes': {
                    'is_alive': actor.is_alive,
                    'is_stationary': speed < 1.0
                }
            }

            return obj_info

        except Exception as e:
            return None

    def _get_object_class(self, type_id):
        """获取物体类别（增强版）"""
        type_lower = type_id.lower()

        if 'vehicle' in type_lower:
            if 'tesla' in type_lower:
                return 'car'
            elif 'audi' in type_lower:
                return 'car'
            elif 'mini' in type_lower:
                return 'car'
            elif 'mercedes' in type_lower:
                return 'car'
            elif 'nissan' in type_lower:
                return 'car'
            elif 'bmw' in type_lower:
                return 'car'
            elif 'truck' in type_lower:
                return 'truck'
            elif 'bus' in type_lower:
                return 'bus'
            else:
                return 'vehicle'

        elif 'walker' in type_lower:
            return 'pedestrian'

        elif 'traffic' in type_lower:
            return 'traffic_light'

        elif 'bike' in type_lower or 'bicycle' in type_lower:
            return 'bicycle'

        elif 'motorcycle' in type_lower:
            return 'motorcycle'

        else:
            return 'unknown'

    def _analyze_pedestrian_risk(self, pedestrians, vehicles):
        """分析行人风险（增强版）"""
        high_risk_count = 0
        near_miss_count = 0
        distances = []

        for pedestrian in pedestrians:
            min_distance = float('inf')
            min_vehicle = None

            for vehicle in vehicles:
                # 计算距离
                p_loc = pedestrian['location']
                v_loc = vehicle['location']

                distance = np.sqrt(
                    (p_loc['x'] - v_loc['x']) ** 2 +
                    (p_loc['y'] - v_loc['y']) ** 2
                )

                # 考虑车辆速度和类型
                vehicle_speed = vehicle['velocity'].get('speed_kmh', 0)
                vehicle_type = vehicle.get('type', 'car')

                # 调整风险阈值（卡车和公交车需要更大的安全距离）
                safe_distance = 5.0
                if vehicle_type in ['truck', 'bus']:
                    safe_distance = 7.0

                # 考虑速度因素
                if vehicle_speed > 30:
                    safe_distance *= (1 + (vehicle_speed - 30) / 50)

                distances.append({
                    'pedestrian_id': pedestrian['id'],
                    'vehicle_id': vehicle['id'],
                    'distance': distance,
                    'vehicle_speed': vehicle_speed,
                    'vehicle_type': vehicle_type,
                    'safe_distance': safe_distance
                })

                if distance < safe_distance:
                    if distance < safe_distance * 0.5:
                        high_risk_count += 1
                    else:
                        near_miss_count += 1

        # 评估整体风险级别
        risk_level = 'low'
        if high_risk_count > 3:
            risk_level = 'critical'
        elif high_risk_count > 1:
            risk_level = 'high'
        elif near_miss_count > 3:
            risk_level = 'medium'

        return {
            'high_risk_count': high_risk_count,
            'near_miss_count': near_miss_count,
            'distances': distances,
            'risk_level': risk_level
        }

    def _track_pedestrian_trajectories(self, frame_num, pedestrians):
        """跟踪行人轨迹"""
        for pedestrian in pedestrians:
            ped_id = pedestrian['id']
            if ped_id not in self.safety_stats['pedestrian_trajectories']:
                self.safety_stats['pedestrian_trajectories'][ped_id] = []

            self.safety_stats['pedestrian_trajectories'][ped_id].append({
                'frame': frame_num,
                'location': pedestrian['location'],
                'velocity': pedestrian['velocity']
            })

            # 只保留最近100帧的轨迹
            if len(self.safety_stats['pedestrian_trajectories'][ped_id]) > 100:
                self.safety_stats['pedestrian_trajectories'][ped_id].pop(0)

    def _detect_risk_zones(self, pedestrians, vehicles):
        """检测高风险区域"""
        risk_zones = []

        # 行人密集区域
        if len(pedestrians) >= 3:
            # 计算行人质心
            locations = np.array([[p['location']['x'], p['location']['y']] for p in pedestrians])
            centroid = np.mean(locations, axis=0)

            # 计算区域半径
            distances = np.sqrt(np.sum((locations - centroid) ** 2, axis=1))
            radius = np.max(distances) + 5.0

            risk_zones.append({
                'type': 'pedestrian_crowd',
                'center': {'x': float(centroid[0]), 'y': float(centroid[1])},
                'radius': float(radius),
                'pedestrian_count': len(pedestrians),
                'risk_level': 'high' if len(pedestrians) > 5 else 'medium'
            })

        # 车辆行人交互区域
        for vehicle in vehicles:
            if vehicle['velocity'].get('speed_kmh', 0) > 30:
                v_loc = vehicle['location']
                nearby_pedestrians = 0

                for pedestrian in pedestrians:
                    p_loc = pedestrian['location']
                    distance = np.sqrt(
                        (p_loc['x'] - v_loc['x']) ** 2 +
                        (p_loc['y'] - v_loc['y']) ** 2
                    )

                    if distance < 20:
                        nearby_pedestrians += 1

                if nearby_pedestrians >= 2:
                    risk_zones.append({
                        'type': 'vehicle_pedestrian_interaction',
                        'location': v_loc,
                        'vehicle_speed': vehicle['velocity'].get('speed_kmh', 0),
                        'pedestrian_count': nearby_pedestrians,
                        'risk_level': 'high' if vehicle['velocity'].get('speed_kmh', 0) > 50 else 'medium'
                    })

        return risk_zones

    def _save_annotations(self, frame_num, annotations):
        """保存标注到文件"""
        filename = f"frame_{frame_num:06d}.json"
        filepath = os.path.join(self.annotations_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        # 同时更新总标注文件
        with self.lock:
            self.frame_annotations[frame_num] = annotations
            self._update_master_annotation()

    def _update_master_annotation(self):
        """更新主标注文件"""
        master_file = os.path.join(self.annotations_dir, "annotations.json")

        # 计算安全统计
        total_high_risk = 0
        total_near_miss = 0
        max_risk_level = 'low'

        for frame_data in self.frame_annotations.values():
            total_high_risk += frame_data['safety_info']['high_risk_interactions']
            total_near_miss += frame_data['safety_info']['near_miss_count']

            # 更新最高风险级别
            risk_level = frame_data['safety_info']['risk_level']
            risk_weights = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
            if risk_weights[risk_level] > risk_weights[max_risk_level]:
                max_risk_level = risk_level

        master_data = {
            'total_frames': len(self.frame_annotations),
            'total_objects': self.object_counter,
            'frames': list(self.frame_annotations.keys()),
            'created': datetime.now().isoformat(),
            'safety_summary': {
                'total_pedestrians': sum(f['safety_info']['pedestrian_count'] for f in self.frame_annotations.values()),
                'total_vehicles': sum(f['safety_info']['vehicle_count'] for f in self.frame_annotations.values()),
                'total_high_risk': total_high_risk,
                'total_near_misses': total_near_miss,
                'max_risk_level': max_risk_level,
                'pedestrian_trajectories': len(self.safety_stats['pedestrian_trajectories'])
            }
        }

        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)

    def generate_summary(self):
        """生成标注摘要（增强版）"""
        vehicle_count = 0
        pedestrian_count = 0
        bicycle_count = 0
        other_count = 0
        high_risk_count = 0
        near_miss_count = 0

        for frame_data in self.frame_annotations.values():
            for obj in frame_data.get('objects', []):
                if obj['class'] in ['car', 'vehicle', 'truck', 'bus']:
                    vehicle_count += 1
                elif obj['class'] == 'pedestrian':
                    pedestrian_count += 1
                elif obj['class'] == 'bicycle':
                    bicycle_count += 1
                else:
                    other_count += 1

            high_risk_count += frame_data['safety_info']['high_risk_interactions']
            near_miss_count += frame_data['safety_info']['near_miss_count']

        # 计算安全指标
        total_interactions = high_risk_count + near_miss_count
        safety_score = 100
        if total_interactions > 0:
            risk_ratio = high_risk_count / total_interactions
            safety_score = max(0, 100 - risk_ratio * 100)

        summary = {
            'total_frames': len(self.frame_annotations),
            'total_objects': self.object_counter,
            'vehicles': vehicle_count,
            'pedestrians': pedestrian_count,
            'bicycles': bicycle_count,
            'other_objects': other_count,
            'high_risk_interactions': high_risk_count,
            'near_miss_interactions': near_miss_count,
            'average_objects_per_frame': self.object_counter / len(
                self.frame_annotations) if self.frame_annotations else 0,
            'safety_metrics': {
                'safety_score': safety_score,
                'pedestrian_to_vehicle_ratio': pedestrian_count / max(1, vehicle_count),
                'high_risk_percentage': high_risk_count / max(1, total_interactions) * 100,
                'interaction_frequency': total_interactions / len(
                    self.frame_annotations) if self.frame_annotations else 0
            },
            'trajectory_analysis': {
                'unique_pedestrians': len(self.safety_stats['pedestrian_trajectories']),
                'average_trajectory_length': np.mean(
                    [len(t) for t in self.safety_stats['pedestrian_trajectories'].values()]) if self.safety_stats[
                    'pedestrian_trajectories'] else 0
            }
        }

        summary_file = os.path.join(self.annotations_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    def generate_safety_report(self):
        """生成详细安全报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'frames_analyzed': len(self.frame_annotations),
            'safety_statistics': self.safety_stats,
            'risk_analysis': self._analyze_risk_patterns(),
            'recommendations': self._generate_safety_recommendations()
        }

        report_file = os.path.join(self.annotations_dir, "safety_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _analyze_risk_patterns(self):
        """分析风险模式"""
        risk_patterns = {
            'high_risk_frames': [],
            'time_of_day_patterns': {},
            'location_patterns': {},
            'speed_patterns': {}
        }

        # 分析高风险帧的模式
        for frame_num, frame_data in self.frame_annotations.items():
            if frame_data['safety_info']['high_risk_interactions'] > 0:
                risk_patterns['high_risk_frames'].append({
                    'frame': frame_num,
                    'high_risk_count': frame_data['safety_info']['high_risk_interactions'],
                    'risk_level': frame_data['safety_info']['risk_level']
                })

        return risk_patterns

    def _generate_safety_recommendations(self):
        """生成安全建议"""
        recommendations = []

        # 基于统计数据的建议
        total_high_risk = sum(f['safety_info']['high_risk_interactions'] for f in self.frame_annotations.values())

        if total_high_risk > 10:
            recommendations.append("建议增加行人安全监控频率")
            recommendations.append("考虑降低高风险区域的车辆速度限制")

        if self.safety_stats['near_misses'] > 20:
            recommendations.append("建议实施主动行人检测和预警系统")
            recommendations.append("考虑增加人行横道和行人信号灯")

        # 基于轨迹分析的建议
        if len(self.safety_stats['pedestrian_trajectories']) > 0:
            avg_speed = np.mean([
                np.sqrt(p['velocity']['x'] ** 2 + p['velocity']['y'] ** 2)
                for traj in self.safety_stats['pedestrian_trajectories'].values()
                for p in traj
            ])

            if avg_speed > 2.0:  # 米/秒
                recommendations.append("行人移动速度较快，建议加强行人流量管理")

        return recommendations