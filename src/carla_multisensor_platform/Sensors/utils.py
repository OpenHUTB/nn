import time
import pygame
from collections import deque

class CustomTimer:
    def __init__(self, window_size=30):
        self.prev_timestamp = None
        self.fps_queue = deque(maxlen=window_size)
        
    def update(self, timestamp):
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            self.prev_time = time.time()
            return 0.0, 0.0, 0.0
        
        elapsed_sim = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp
        
        now = time.time()
        elapsed_real = now - self.prev_time
        self.prev_time = now
        
        instant_fps, avg_fps = self.instant_FPS(elapsed_sim, elapsed_real)
        return elapsed_sim, instant_fps, avg_fps
        
    def instant_FPS(self, elapsed_sim, elapsed_real):
        instant_fps = 1.0 / elapsed_sim if elapsed_sim > 0 else 0.0
        self.fps_queue.append(instant_fps)
        avg_fps = sum(self.fps_queue) / len(self.fps_queue)
        
        return instant_fps, avg_fps
    
    def timer_callback(self, sensor_name, timer):
        elapsed, instant_fps, avg_fps = self.update(timer)
        # print(f"[{sensor_name}] Δt={elapsed:.3f}s | FPS={instant_fps:.1f} | Avg FPS={avg_fps:.1f}")
        