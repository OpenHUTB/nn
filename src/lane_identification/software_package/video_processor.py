"""
视频处理模块 - 负责视频和摄像头处理
添加帧缓冲和自适应跳帧优化
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
from typing import Optional, Callable
import sys


class VideoProcessor:
    """视频处理器 - 优化版本"""
    
    def __init__(self, config):
        self.config = config
        self.video_capture = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = None
        self.frame_count = 0
        self.fps = config.video_fps
        self.frame_skip = config.video_frame_skip
        self.processor_thread = None
        
        # 帧缓冲（新增）
        self.frame_buffer = deque(maxlen=config.frame_buffer_size if hasattr(config, 'frame_buffer_size') else 5)
        self.buffer_lock = threading.Lock()
        self.frame_producer_thread = None
        self.stop_producer = threading.Event()
        
        # 自适应跳帧（新增）
        self.adaptive_skip_frames = getattr(config, 'adaptive_skip_frames', True)
        self.max_processing_time = getattr(config, 'max_processing_time', 0.1)
        self.min_fps = getattr(config, 'min_fps', 5)
        self.last_processing_time = 0
        self.adaptive_frame_skip = 1
        
        # 性能统计
        self.frame_times = []
        self.processing_times = []
        self.frame_drop_count = 0
        self.total_frames_processed = 0
        
        # 错误处理
        self.error_count = 0
        self.max_retries = 3
        
    def open_video_file(self, video_path):
        """打开视频文件"""
        try:
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                print(f"无法打开视频文件: {video_path}")
                return False
            
            # 获取视频信息
            self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = self.config.video_fps
            
            # 重置状态
            self.reset_stats()
            
            print(f"视频已打开: {video_path}")
            print(f"帧数: {self.frame_count}, FPS: {self.fps}")
            
            return True
            
        except Exception as e:
            print(f"打开视频文件失败: {e}")
            return False
    
    def open_camera(self, camera_id=None):
        """打开摄像头（修复版）"""
        try:
            # 如果提供了camera_id则使用，否则使用配置中的ID
            if camera_id is not None:
                cam_id = camera_id
            elif hasattr(self.config, 'camera_id'):
                cam_id = self.config.camera_id
            else:
                cam_id = 0  # 默认摄像头索引
            
            print(f"正在尝试打开摄像头 {cam_id}...")
            
            # 尝试不同的后端
            backends_to_try = [
                (cam_id, cv2.CAP_DSHOW),    # Windows DirectShow
                (cam_id, cv2.CAP_MSMF),     # Windows Media Foundation
                (cam_id, cv2.CAP_ANY),      # 自动选择
            ]
            
            for backend in backends_to_try:
                try:
                    self.video_capture = cv2.VideoCapture(*backend)
                    if self.video_capture.isOpened():
                        # 测试读取一帧
                        ret, test_frame = self.video_capture.read()
                        if ret:
                            print(f"摄像头 {cam_id} 已成功打开 (后端: {backend[1]})")
                            
                            # 获取摄像头参数
                            actual_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                            
                            self.fps = actual_fps if actual_fps > 0 else 30
                            
                            # 重置状态
                            self.reset_stats()
                            
                            print(f"分辨率: {actual_width}x{actual_height}")
                            print(f"FPS: {self.fps}")
                            
                            # 尝试设置合理的参数
                            try:
                                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.video_capture.set(cv2.CAP_PROP_FPS, 30)
                                self.video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                                self.video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                            except:
                                pass  # 忽略参数设置错误
                            
                            return True
                        else:
                            self.video_capture.release()
                except Exception as e:
                    print(f"尝试后端 {backend[1]} 失败: {e}")
                    continue
            
            print(f"无法打开摄像头 {cam_id}，所有后端尝试失败")
            return False
            
        except Exception as e:
            print(f"打开摄像头失败: {e}")
            return False
    
    def start_processing(self, callback):
        """开始视频处理"""
        if self.video_capture is None:
            print("错误：未打开视频源")
            return False
        
        self.is_playing = True
        self.is_paused = False
        self.stop_producer.clear()
        
        # 如果启用了帧缓冲，启动生产者线程
        if getattr(self.config, 'enable_frame_buffer', True):
            self.frame_producer_thread = threading.Thread(
                target=self._frame_producer,
                daemon=True
            )
            self.frame_producer_thread.start()
            print("帧缓冲生产者线程已启动")
        
        # 启动处理线程
        self.processor_thread = threading.Thread(
            target=self._process_frames_optimized,
            args=(callback,),
            daemon=True
        )
        self.processor_thread.start()
        
        print("视频处理已开始")
        return True
    
    def _frame_producer(self):
        """帧生产者线程 - 填充帧缓冲"""
        while self.is_playing and not self.stop_producer.is_set():
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            # 检查缓冲是否已满
            with self.buffer_lock:
                if len(self.frame_buffer) >= self.frame_buffer.maxlen:
                    time.sleep(0.01)
                    continue
            
            # 读取帧
            ret, frame = self.video_capture.read()
            if not ret:
                # 视频结束，等待一段时间后重试
                time.sleep(0.5)
                if not self.camera_mode:
                    break
                continue
            
            # 添加到缓冲
            with self.buffer_lock:
                self.frame_buffer.append(frame)
    
    def _process_frames_optimized(self, callback):
        """优化版的视频帧处理"""
        frame_number = 0
        last_time = time.time()
        skip_counter = 0
        
        while self.is_playing and self.video_capture is not None:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            # 更新自适应跳帧
            self._update_adaptive_skip()
            
            # 自适应跳帧逻辑
            skip_counter += 1
            if skip_counter < self.adaptive_frame_skip:
                # 跳过帧，但保持进度
                if not self.camera_mode:
                    self.video_capture.grab()
                continue
            skip_counter = 0
            
            # 获取帧
            frame = None
            if getattr(self.config, 'enable_frame_buffer', True) and self.frame_buffer:
                with self.buffer_lock:
                    if self.frame_buffer:
                        frame = self.frame_buffer.popleft()
            else:
                # 直接读取
                ret, frame = self.video_capture.read()
                if not ret:
                    print("视频结束或读取失败")
                    break
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_number += 1
            self.total_frames_processed += 1
            
            # 记录开始时间
            start_time = time.time()
            
            # 准备帧信息
            frame_info = {
                'frame_number': frame_number,
                'frame_time': start_time,
                'fps': self.fps,
                'frame_skip': self.adaptive_frame_skip,
                'buffer_size': len(self.frame_buffer)
            }
            
            # 调用回调函数处理帧
            try:
                callback(frame, frame_info)
            except Exception as e:
                print(f"帧处理回调失败: {e}")
                self.error_count += 1
                if self.error_count > self.max_retries:
                    print("错误次数过多，停止处理")
                    break
            
            # 计算处理时间
            processing_time = time.time() - start_time
            self.last_processing_time = processing_time
            
            # 记录处理时间
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 10:
                self.processing_times.pop(0)
            
            # 计算实际FPS
            current_time = time.time()
            if current_time - last_time >= 1.0:
                processed_frames = frame_number
                if processed_frames > 0:
                    actual_fps = processed_frames / (current_time - last_time)
                    self.fps = actual_fps
                    print(f"实际FPS: {actual_fps:.1f}, 跳帧: {self.adaptive_frame_skip}, "
                          f"缓冲: {len(self.frame_buffer)}/{self.frame_buffer.maxlen}")
                last_time = current_time
                frame_number = 0
            
            # 控制处理速度（仅限视频文件）
            if not self.camera_mode:
                target_delay = 1.0 / self.fps
                if processing_time < target_delay:
                    sleep_time = target_delay - processing_time
                    if sleep_time > 0.001:  # 只休眠有意义的时间
                        time.sleep(sleep_time)
    
    def _update_adaptive_skip(self):
        """更新自适应跳帧参数"""
        if not self.adaptive_skip_frames:
            self.adaptive_frame_skip = self.frame_skip
            return
        
        # 基于最近的处理时间调整跳帧
        if len(self.processing_times) >= 3:
            avg_processing_time = np.mean(self.processing_times[-3:])
            
            if avg_processing_time > self.max_processing_time:
                # 处理时间过长，增加跳帧
                new_skip = min(10, self.adaptive_frame_skip + 1)
                if new_skip != self.adaptive_frame_skip:
                    self.adaptive_frame_skip = new_skip
                    print(f"增加跳帧到 {self.adaptive_frame_skip} (处理时间: {avg_processing_time:.3f}s)")
            elif avg_processing_time < self.max_processing_time * 0.5:
                # 处理时间充足，减少跳帧
                new_skip = max(1, self.adaptive_frame_skip - 1)
                if new_skip != self.adaptive_frame_skip:
                    self.adaptive_frame_skip = new_skip
                    print(f"减少跳帧到 {self.adaptive_frame_skip} (处理时间: {avg_processing_time:.3f}s)")
    
    def pause(self):
        """暂停视频处理"""
        self.is_paused = True
        print("视频已暂停")
    
    def resume(self):
        """恢复视频处理"""
        self.is_paused = False
        print("视频已恢复")
    
    def stop(self):
        """停止视频处理"""
        self.is_playing = False
        self.is_paused = False
        self.stop_producer.set()
        
        # 等待线程结束
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        
        if self.frame_producer_thread and self.frame_producer_thread.is_alive():
            self.frame_producer_thread.join(timeout=1.0)
        
        print("视频处理已停止")
    
    def release(self):
        """释放资源"""
        self.stop()
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        # 清空缓冲
        with self.buffer_lock:
            self.frame_buffer.clear()
        
        print("视频资源已释放")
    
    def get_frame(self):
        """获取当前帧"""
        if self.video_capture is None:
            return None
        
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            return frame
        
        return None
    
    def get_video_info(self):
        """获取视频信息"""
        if self.video_capture is None:
            return {}
        
        return {
            'fps': self.fps,
            'frame_width': int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'frame_height': int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': self.frame_count,
            'is_playing': self.is_playing,
            'is_paused': self.is_paused,
            'adaptive_frame_skip': self.adaptive_frame_skip,
            'buffer_usage': f"{len(self.frame_buffer)}/{self.frame_buffer.maxlen}",
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'frame_drop_rate': self.frame_drop_count / max(self.total_frames_processed, 1)
        }
    
    def set_frame_position(self, position):
        """设置帧位置"""
        if self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            # 清空缓冲
            with self.buffer_lock:
                self.frame_buffer.clear()
    
    def reset_stats(self):
        """重置统计信息"""
        self.frame_times.clear()
        self.processing_times.clear()
        self.frame_drop_count = 0
        self.total_frames_processed = 0
        self.error_count = 0
        with self.buffer_lock:
            self.frame_buffer.clear()
    
    @property
    def camera_mode(self):
        """是否处于摄像头模式"""
        return self.video_capture is not None and not hasattr(self, '_is_file') or \
               (hasattr(self, '_is_file') and not self._is_file)