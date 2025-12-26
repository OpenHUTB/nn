import json
import os
import numpy as np
from datetime import datetime


class AnnotationGenerator:
    """标注生成器"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(self.annotations_dir, exist_ok=True)

        self.frame_annotations = {}
        self.object_counter = 0

    def detect_objects(self, world, frame_num, timestamp):
        """检测场景中的物体"""
        annotations = {
            'frame_id': frame_num,
            'timestamp': timestamp,
            'objects': [],
            'camera_info': {},
            'safety_info': {
                'pedestrian_count': 0,
                'vehicle_count': 0,
                'high_risk_interactions': 0
            }
        }

        try:
            actors = world.get_actors()

            for actor in actors:
                obj_type = actor.type_id

                if 'vehicle' in obj_type or 'walker' in obj_type:
                    obj_info = self._extract_object_info(actor)
                    if obj_info:
                        annotations['objects'].append(obj_info)
                        self.object_counter += 1

                        # 更新安全统计
                        if 'walker' in obj_type:
                            annotations['safety_info']['pedestrian_count'] += 1
                        elif 'vehicle' in obj_type:
                            annotations['safety_info']['vehicle_count'] += 1

            # 检测高风险交互
            annotations['safety_info']['high_risk_interactions'] = self._detect_high_risk_interactions(
                annotations['objects'])

            self._save_annotations(frame_num, annotations)
            return annotations

        except Exception as e:
            print(f"物体检测失败: {e}")
            return annotations

    def _extract_object_info(self, actor):
        """提取物体信息"""
        try:
            bbox = actor.bounding_box
            location = actor.get_location()
            velocity = actor.get_velocity()

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
                    'z': float(velocity.z)
                },
                'bounding_box': {
                    'extent': {
                        'x': float(bbox.extent.x),
                        'y': float(bbox.extent.y),
                        'z': float(bbox.extent.z)
                    }
                },
                'attributes': {
                    'is_alive': actor.is_alive
                }
            }

            return obj_info

        except Exception as e:
            return None

    def _get_object_class(self, type_id):
        """获取物体类别"""
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
            else:
                return 'vehicle'

        elif 'walker' in type_lower:
            return 'pedestrian'

        elif 'traffic' in type_lower:
            return 'traffic_light'

        else:
            return 'unknown'

    def _detect_high_risk_interactions(self, objects):
        """检测高风险交互"""
        high_risk_count = 0
        pedestrians = [obj for obj in objects if obj['class'] == 'pedestrian']
        vehicles = [obj for obj in objects if obj['class'] in ['car', 'vehicle']]

        for pedestrian in pedestrians:
            for vehicle in vehicles:
                # 计算距离
                p_loc = pedestrian['location']
                v_loc = vehicle['location']

                distance = np.sqrt(
                    (p_loc['x'] - v_loc['x']) ** 2 +
                    (p_loc['y'] - v_loc['y']) ** 2 +
                    (p_loc['z'] - v_loc['z']) ** 2
                )

                # 如果距离小于5米，认为是高风险交互
                if distance < 5.0:
                    high_risk_count += 1

        return high_risk_count

    def _save_annotations(self, frame_num, annotations):
        """保存标注到文件"""
        filename = f"frame_{frame_num:06d}.json"
        filepath = os.path.join(self.annotations_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        # 同时更新总标注文件
        self.frame_annotations[frame_num] = annotations
        self._update_master_annotation()

    def _update_master_annotation(self):
        """更新主标注文件"""
        master_file = os.path.join(self.annotations_dir, "annotations.json")

        master_data = {
            'total_frames': len(self.frame_annotations),
            'total_objects': self.object_counter,
            'frames': list(self.frame_annotations.keys()),
            'created': datetime.now().isoformat(),
            'safety_summary': {
                'total_pedestrians': sum(f['safety_info']['pedestrian_count'] for f in self.frame_annotations.values()),
                'total_vehicles': sum(f['safety_info']['vehicle_count'] for f in self.frame_annotations.values()),
                'total_high_risk': sum(
                    f['safety_info']['high_risk_interactions'] for f in self.frame_annotations.values())
            }
        }

        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)

    def generate_summary(self):
        """生成标注摘要"""
        vehicle_count = 0
        pedestrian_count = 0
        other_count = 0
        high_risk_count = 0

        for frame_data in self.frame_annotations.values():
            for obj in frame_data.get('objects', []):
                if obj['class'] in ['car', 'vehicle']:
                    vehicle_count += 1
                elif obj['class'] == 'pedestrian':
                    pedestrian_count += 1
                else:
                    other_count += 1

            high_risk_count += frame_data['safety_info']['high_risk_interactions']

        summary = {
            'total_frames': len(self.frame_annotations),
            'total_objects': self.object_counter,
            'vehicles': vehicle_count,
            'pedestrians': pedestrian_count,
            'other_objects': other_count,
            'high_risk_interactions': high_risk_count,
            'average_objects_per_frame': self.object_counter / len(
                self.frame_annotations) if self.frame_annotations else 0,
            'safety_metrics': {
                'pedestrian_to_vehicle_ratio': pedestrian_count / max(1, vehicle_count),
                'high_risk_percentage': high_risk_count / max(1, self.object_counter) * 100
            }
        }

        summary_file = os.path.join(self.annotations_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary