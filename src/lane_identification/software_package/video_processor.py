"""
视频处理模块 - 负责视频和摄像头处理
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Dict, Any
from config import AppConfig

class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.video_capture = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = None
        self.frame_count = 0
        self.fps = 0
        self.frame_skip = config.video_frame_skip
        self.processor_thread = None
        
        # 性能统计
        self.frame_times = []
        self.processing_times = []
        
        print("视频处理器已初始化")
    
    def open_video_file(self, video_path: str) -> bool:
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
            
            print(f"视频已打开: {video_path}")
            print(f"帧数: {self.frame_count}, FPS: {self.fps}")
            
            return True
            
        except Exception as e:
            print(f"打开视频文件失败: {e}")
            return False
    
    def open_camera(self, camera_id: Optional[int] = None) -> bool:
        """打开摄像头"""
        try:
            cam_id = camera_id if camera_id is not None else self.config.camera_id
            self.video_capture = cv2.VideoCapture(cam_id)
            
            if not self.video_capture.isOpened():
                print(f"无法打开摄像头 {cam_id}")
                return False
            
            # 设置摄像头参数
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_capture.set(cv2.CAP_PROP_FPS, self.config.video_fps)
            
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = self.config.video_fps
            
            print(f"摄像头已打开: {cam_id}")
            print(f"FPS: {self.fps}")
            
            return True
            
        except Exception as e:
            print(f"打开摄像头失败: {e}")
            return False
    
    def start_processing(self, callback: callable):
        """开始视频处理"""
        if self.video_capture is None:
            print("错误：未打开视频源")
            return False
        
        self.is_playing = True
        self.is_paused = False
        
        # 启动处理线程
        self.processor_thread = threading.Thread(
            target=self._process_frames,
            args=(callback,),
            daemon=True
        )
        self.processor_thread.start()
        
        print("视频处理已开始")
        return True
    
    def _process_frames(self, callback: callable):
        """处理视频帧"""
        frame_number = 0
        skip_counter = 0
        last_time = time.time()
        
        while self.is_playing and self.video_capture is not None:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            # 跳过一些帧以提高性能
            skip_counter += 1
            if skip_counter < self.frame_skip:
                # 仍然需要读取帧以保持进度
                self.video_capture.grab()
                continue
            skip_counter = 0
            
            # 读取帧
            ret, frame = self.video_capture.read()
            if not ret:
                print("视频结束或读取失败")
                break
            
            frame_number += 1
            
            # 记录开始时间
            start_time = time.time()
            
            # 准备帧信息
            frame_info = {
                'frame_number': frame_number,
                'frame_time': start_time,
                'fps': self.fps
            }
            
            # 调用回调函数处理帧
            try:
                callback(frame, frame_info)
            except Exception as e:
                print(f"帧处理回调失败: {e}")
            
            # 计算处理时间
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 10:
                self.processing_times.pop(0)
            
            # 计算实际FPS
            current_time = time.time()
            if current_time - last_time >= 1.0:
                self.fps = frame_number / (current_time - last_time)
                last_time = current_time
                frame_number = 0
            
            # 控制处理速度
            target_delay = 1.0 / self.fps
            if processing_time < target_delay:
                time.sleep(target_delay - processing_time)
    
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
        
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        
        print("视频处理已停止")
    
    def release(self):
        """释放资源"""
        self.stop()
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
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
    
    def get_video_info(self) -> Dict[str, Any]:
        """获取视频信息"""
        if self.video_capture is None:
            return {}
        
        return {
            'fps': self.fps,
            'frame_width': int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'frame_height': int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': self.frame_count,
            'is_playing': self.is_playing,
            'is_paused': self.is_paused
        }
    
    def set_frame_position(self, position: int):
        """设置帧位置"""
        if self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)