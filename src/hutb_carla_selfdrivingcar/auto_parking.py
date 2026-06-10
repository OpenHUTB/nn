import carla
import math

class AutoParking:
    """自动泊车系统"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.parking_speed = 5.0  # km/h
        self.reverse_speed = 3.0  # km/h
        self.parking_complete = False
        
    def get_vehicle_speed(self):
        """获取当前速度（km/h）"""
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    def find_parking_spot(self):
        """模拟寻找停车位"""
        vehicle_loc = self.vehicle.get_location()
        
        # 模拟检测：随机找到一个空位
        # 实际项目中应该使用传感器检测
        return True
    
    def park(self):
        """执行泊车"""
        if self.parking_complete:
            return True
            
        stage = self.get_parking_stage()
        
        if stage == "寻找车位":
            if self.find_parking_spot():
                return "靠近车位"
                
        elif stage == "靠近车位":
            # 缓慢靠近车位
            throttle = 0.1
            brake = 0.0
            steer = 0.0
            self.apply_control(throttle, brake, steer)
            
            if self.is_close_to_spot():
                return "倒车入库"
                
        elif stage == "倒车入库":
            # 倒车入库
            throttle = 0.0
            brake = 0.0
            steer = -1.0  # 方向盘左打满
            self.apply_control(throttle, brake, steer)
            
            if self.is_reversed_enough():
                return "调整位置"
                
        elif stage == "调整位置":
            # 微调车身
            throttle = 0.05
            brake = 0.0
            steer = 0.0
            self.apply_control(throttle, brake, steer)
            
            if self.is_parked():
                self.parking_complete = True
                return "完成"
                
        return stage
    
    def get_parking_stage(self):
        """获取当前泊车阶段"""
        if self.parking_complete:
            return "完成"
        
        # 简单的状态机实现
        if not hasattr(self, '_stage'):
            self._stage = "寻找车位"
        return self._stage
    
    def set_stage(self, stage):
        """设置泊车阶段"""
        self._stage = stage
    
    def is_close_to_spot(self):
        """是否靠近车位"""
        return True
    
    def is_reversed_enough(self):
        """是否倒够了"""
        return True
    
    def is_parked(self):
        """是否停好"""
        return True
    
    def apply_control(self, throttle, brake, steer):
        """应用控制"""
        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        )
        self.vehicle.apply_control(control)
    
    def stop(self):
        """停止"""
        control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        self.vehicle.apply_control(control)
