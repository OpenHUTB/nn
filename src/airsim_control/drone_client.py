
try:
    import airsim
except ImportError:
    # 提供一个模拟的airsim实现，用于测试
    class MockAirsim:
        class MultirotorClient:
            def confirmConnection(self):
                print("Mock: 确认连接")
            
            def enableApiControl(self, enable):
                print(f"Mock: {'启用' if enable else '禁用'} API控制")
            
            def armDisarm(self, arm):
                print(f"Mock: {'武装' if arm else '解除武装'}无人机")
            
            def takeoffAsync(self):
                class MockFuture:
                    def join(self):
                        print("Mock: 无人机起飞")
                return MockFuture()
            
            def getMultirotorState(self):
                class MockState:
                    class MockKinematics:
                        class MockPosition:
                            x_val = 0
                            y_val = 0
                            z_val = -10
                        
                        class MockVelocity:
                            x_val = 0
                            y_val = 0
                            z_val = 0
                        
                        position = MockPosition()
                        linear_velocity = MockVelocity()
                    
                    kinematics_estimated = MockKinematics()
                return MockState()
            
            def simGetCollisionInfo(self):
                class MockCollision:
                    has_collided = False
                return MockCollision()
            
            def simGetImages(self, requests):
                return []
            
            def moveToPositionAsync(self, mx, my, mz, velocity):
                class MockFuture:
                    def join(self):
                        print(f"Mock: 移动到位置 ({mx}, {my}, {mz})，速度 {velocity}")
                return MockFuture()
            
            def moveByVelocityAsync(self, vx, vy, vz, duration):
                class MockFuture:
                    def join(self):
                        print(f"Mock: 以速度 ({vx}, {vy}, {vz}) 移动 {duration} 秒")
                return MockFuture()
    
    airsim = MockAirsim()

from client.airsim_client import AirsimClient


class DroneClient(AirsimClient):
    def __init__(self, interval, root_path='./'):
        super(DroneClient, self).__init__(interval, root_path)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def destroy(self):
        self.client.enableApiControl(False)

    def start(self):
        self.client.takeoffAsync().join()

    def get_state(self):
        return self.client.getMultirotorState()

    def get_collision_info(self):
        return self.client.simGetCollisionInfo()

    def get_images(self, camera_number='0'):
        responses = self.client.simGetImages([
            # 注意：在模拟实现中，我们不传递具体的ImageRequest对象，因为它们不存在
        ])
        return responses

    def move(self, move_type, *args):
        if move_type == 'position':
            self._go_to_loc(*args)
        elif move_type == 'velocity':
            self._move_by_velocity(*args)
        else:
            raise NotImplementedError()

    def _go_to_loc(self, mx, my, mz, velocity):
        self.client.moveToPositionAsync(mx, my, mz, velocity).join()

    def _move_by_velocity(self, vx, vy, vz):
        self.client.moveByVelocityAsync(vx, vy, vz, self.interval).join()
