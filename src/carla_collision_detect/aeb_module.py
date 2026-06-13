import time

class AEBController:
    def __init__(self):
        self.x_history = []
        self.time_history = []
        self.aeb_trigger_time = 0.0  

    def evaluate(self, current_dist, target_class, target_x, is_changing_lane):
        current_time = time.time()

        # 1. 锁死机制：一旦触发急刹，强行抱死 4.0 秒！(保证彻底停稳)
        if current_time - self.aeb_trigger_time < 4.0:
            return True

        # 2. 清理超过 0.5 秒的陈旧数据，同时保留偶尔漏检的断点记忆
        while self.time_history and current_time - self.time_history[0] > 0.5:
            self.x_history.pop(0)
            self.time_history.pop(0)

        trigger_aeb = False

        if target_class == "person" and target_x is not None:
            self.x_history.append(target_x)
            self.time_history.append(current_time)

            # 3. 只要积攒了 3 帧数据，立刻计算瞬时移动趋势，极速响应
            if len(self.x_history) >= 3:
                dx = self.x_history[-1] - self.x_history[0]
                dt = self.time_history[-1] - self.time_history[0]

                if dt > 0:
                    pixel_speed = dx / dt  

                    if pixel_speed < -15.0 and current_dist < 10.0:
                        if not is_changing_lane:
                            print(f"\033[91m⚠️ 发现行人，自动刹车 (距离: {current_dist:.1f}m)\033[0m")
                            self.aeb_trigger_time = current_time  
                            trigger_aeb = True

        return trigger_aeb