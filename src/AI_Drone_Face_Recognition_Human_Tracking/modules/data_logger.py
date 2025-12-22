# modules/data_logger.py
import json
import time
import datetime
import os
import threading
from collections import deque


class DataLogger:
    def __init__(self, enabled=True, max_records=1000, auto_save_interval=60):
        self.enabled = enabled
        self.max_records = max_records
        self.auto_save_interval = auto_save_interval
        self.records = deque(maxlen=max_records)
        self.last_save_time = time.time()
        self.log_file = None
        self.running = False
        self.save_thread = None

        # 创建日志目录
        self.log_dir = "flight_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.enabled:
            self.start()

    def start(self):
        """启动数据记录"""
        if self.running:
            return

        self.running = True

        # 创建新的日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"flight_{timestamp}.json")

        # 启动自动保存线程
        self.save_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
        self.save_thread.start()

        print(f"✅ 数据记录已启动，日志文件: {self.log_file}")

    def stop(self):
        """停止数据记录"""
        self.running = False
        if self.save_thread:
            self.save_thread.join(timeout=2)

        # 保存剩余数据
        self.save_to_file()
        print("✅ 数据记录已停止")

    def _auto_save_worker(self):
        """自动保存工作线程"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_save_time >= self.auto_save_interval:
                self.save_to_file()
                self.last_save_time = current_time
            time.sleep(5)

    def log_drone_state(self, position, yaw, is_flying, mode):
        """记录无人机状态"""
        if not self.enabled:
            return

        record = {
            'timestamp': time.time(),
            'type': 'drone_state',
            'position': position,
            'yaw': yaw,
            'is_flying': is_flying,
            'mode': mode,
            'battery': 100.0,  # 模拟电池电量
            'signal_strength': 5  # 模拟信号强度
        }
        self.records.append(record)

    def log_detection_result(self, face_count, person_count, recognized_person):
        """记录检测结果"""
        if not self.enabled:
            return

        record = {
            'timestamp': time.time(),
            'type': 'detection',
            'face_count': face_count,
            'person_count': person_count,
            'recognized_person': recognized_person
        }
        self.records.append(record)

    def log_control_action(self, action, params=None):
        """记录控制动作"""
        if not self.enabled:
            return

        record = {
            'timestamp': time.time(),
            'type': 'control',
            'action': action,
            'params': params or {}
        }
        self.records.append(record)

    def log_system_event(self, event_type, message):
        """记录系统事件"""
        if not self.enabled:
            return

        record = {
            'timestamp': time.time(),
            'type': 'system',
            'event': event_type,
            'message': message
        }
        self.records.append(record)

    def save_to_file(self, filename=None):
        """保存数据到文件"""
        if not self.records:
            return False

        try:
            save_file = filename or self.log_file
            if not save_file:
                return False

            # 转换deque为list
            records_list = list(self.records)

            # 添加文件头信息
            data = {
                'metadata': {
                    'created_at': datetime.datetime.now().isoformat(),
                    'total_records': len(records_list),
                    'record_types': set(record['type'] for record in records_list),
                    'duration': records_list[-1]['timestamp'] - records_list[0]['timestamp'] if len(
                        records_list) > 1 else 0
                },
                'records': records_list
            }

            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"✅ 已保存 {len(records_list)} 条记录到: {save_file}")
            return True

        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False

    def load_from_file(self, filename):
        """从文件加载数据"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 清空当前记录
            self.records.clear()

            # 加载记录
            for record in data.get('records', []):
                self.records.append(record)

            print(f"✅ 已从 {filename} 加载 {len(self.records)} 条记录")
            return True

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False

    def get_statistics(self):
        """获取统计信息"""
        if not self.records:
            return {}

        records_list = list(self.records)

        stats = {
            'total_records': len(records_list),
            'first_timestamp': records_list[0]['timestamp'] if records_list else 0,
            'last_timestamp': records_list[-1]['timestamp'] if records_list else 0,
            'duration': records_list[-1]['timestamp'] - records_list[0]['timestamp'] if len(records_list) > 1 else 0,
            'record_types': {},
            'flight_time': 0,
            'total_distance': 0.0,
            'max_altitude': 0.0
        }

        # 统计记录类型
        for record in records_list:
            record_type = record['type']
            stats['record_types'][record_type] = stats['record_types'].get(record_type, 0) + 1

        # 计算飞行数据
        flight_start = None
        last_position = None

        for record in records_list:
            if record['type'] == 'drone_state':
                position = record['position']

                # 计算飞行时间
                if record['is_flying']:
                    if flight_start is None:
                        flight_start = record['timestamp']
                else:
                    if flight_start is not None:
                        stats['flight_time'] += record['timestamp'] - flight_start
                        flight_start = None

                # 计算飞行距离
                if last_position:
                    dx = position[0] - last_position[0]
                    dy = position[1] - last_position[1]
                    dz = position[2] - last_position[2]
                    distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
                    stats['total_distance'] += distance

                    # 更新最大高度
                    if position[2] > stats['max_altitude']:
                        stats['max_altitude'] = position[2]

                last_position = position

        # 处理最后的飞行时间
        if flight_start is not None and records_list:
            stats['flight_time'] += records_list[-1]['timestamp'] - flight_start

        return stats

    def export_to_csv(self, filename=None):
        """导出数据到CSV"""
        if not self.records:
            return False

        try:
            import csv

            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.log_dir, f"flight_export_{timestamp}.csv")

            records_list = list(self.records)

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 写入标题
                writer.writerow([
                    '时间戳', '记录类型', 'X位置', 'Y位置', 'Z位置', '航向',
                    '飞行状态', '控制模式', '人脸数', '行人数', '识别结果'
                ])

                # 写入数据
                for record in records_list:
                    row = [
                        datetime.datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        record['type']
                    ]

                    if record['type'] == 'drone_state':
                        row.extend([
                            record['position'][0],
                            record['position'][1],
                            record['position'][2],
                            record['yaw'],
                            '是' if record['is_flying'] else '否',
                            record['mode'],
                            '', '', ''  # 检测相关字段为空
                        ])
                    elif record['type'] == 'detection':
                        row.extend([
                            '', '', '', '', '', '',  # 无人机相关字段为空
                            record['face_count'],
                            record['person_count'],
                            record['recognized_person']
                        ])
                    else:
                        row.extend(['', '', '', '', '', '', '', '', ''])

                    writer.writerow(row)

            print(f"✅ 数据已导出到CSV: {filename}")
            return True

        except ImportError:
            print("❌ CSV导出失败: 未安装csv模块")
        except Exception as e:
            print(f"❌ CSV导出失败: {e}")

        return False

    def replay_data(self, speed=1.0, callback=None):
        """回放数据"""
        if not self.records:
            print("⚠️  无数据可回放")
            return False

        try:
            print(f"🎬 开始数据回放，速度: {speed}x")
            records_list = list(self.records)

            start_time = records_list[0]['timestamp']
            current_time = start_time

            for i, record in enumerate(records_list):
                # 计算等待时间
                if i > 0:
                    time_diff = (record['timestamp'] - records_list[i - 1]['timestamp']) / speed
                    if time_diff > 0:
                        time.sleep(time_diff)

                # 调用回调函数处理记录
                if callback:
                    callback(record, i, len(records_list))

                current_time = record['timestamp']

            print("✅ 数据回放完成")
            return True

        except KeyboardInterrupt:
            print("⏸️  数据回放被中断")
        except Exception as e:
            print(f"❌ 数据回放失败: {e}")

        return False