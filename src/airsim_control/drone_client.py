from typing import Any

try:
    import airsim
except ImportError:
    # Provide a mock airsim implementation for testing
    class MockAirsim:
        class MultirotorClient:
            def confirmConnection(self) -> None:
                print("Mock: 确认连接")

            def enableApiControl(self, enable: bool) -> None:
                print(f"Mock: {'启用' if enable else '禁用'} API控制")

            def armDisarm(self, arm: bool) -> None:
                print(f"Mock: {'武装' if arm else '解除武装'}无人机")

            def takeoffAsync(self) -> Any:
                class MockFuture:
                    def join(self) -> None:
                        print("Mock: 无人机起飞")
                return MockFuture()

            def getMultirotorState(self) -> Any:
                class MockState:
                    class MockKinematics:
                        class MockPosition:
                            x_val: float = 0
                            y_val: float = 0
                            z_val: float = -10

                        class MockVelocity:
                            x_val: float = 0
                            y_val: float = 0
                            z_val: float = 0

                        position = MockPosition()
                        linear_velocity = MockVelocity()

                    kinematics_estimated = MockKinematics()
                return MockState()

            def simGetCollisionInfo(self) -> Any:
                class MockCollision:
                    has_collided: bool = False
                return MockCollision()

            def simGetImages(self, requests: list) -> list:
                return []

            def moveToPositionAsync(self, mx: float, my: float,
                                   mz: float, velocity: float) -> Any:
                class MockFuture:
                    def join(self) -> None:
                        print(f"Mock: 移动到位置 ({mx}, {my}, {mz})，速度 {velocity}")
                return MockFuture()

            def moveByVelocityAsync(self, vx: float, vy: float,
                                   vz: float, duration: float) -> Any:
                class MockFuture:
                    def join(self) -> None:
                        print(f"Mock: 以速度 ({vx}, {vy}, {vz}) 移动 {duration} 秒")
                return MockFuture()

    airsim = MockAirsim()

from client.airsim_client import AirsimClient


class DroneClient(AirsimClient):
    """AirSim drone client for controlling multirotor vehicles.

    Args:
        interval: Control loop interval in seconds.
        root_path: Root path for data storage.
    """

    def __init__(self, interval: float, root_path: str = './') -> None:
        super(DroneClient, self).__init__(interval, root_path)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def destroy(self) -> None:
        """Disable API control and cleanup."""
        self.client.enableApiControl(False)

    def start(self) -> None:
        """Start the drone by taking off."""
        self.client.takeoffAsync().join()

    def get_state(self) -> Any:
        """Get current drone state.

        Returns:
            Multirotor state object with kinematics information.
        """
        return self.client.getMultirotorState()

    def get_collision_info(self) -> Any:
        """Get collision information.

        Returns:
            Collision info object.
        """
        return self.client.simGetCollisionInfo()

    def get_images(self, camera_number: str = '0') -> list:
        """Get images from drone cameras.

        Args:
            camera_number: Camera identifier.

        Returns:
            List of image responses.
        """
        responses = self.client.simGetImages([])
        return responses

    def move(self, move_type: str, *args: float) -> None:
        """Execute drone movement.

        Args:
            move_type: Movement type ('position' or 'velocity').
            *args: Movement parameters.

        Raises:
            NotImplementedError: If move_type is not supported.
        """
        if move_type == 'position':
            self._go_to_loc(*args)
        elif move_type == 'velocity':
            self._move_by_velocity(*args)
        else:
            raise NotImplementedError(f"Unsupported move type: {move_type}")

    def _go_to_loc(self, mx: float, my: float, mz: float,
                   velocity: float) -> None:
        """Move drone to specified position.

        Args:
            mx: Target X coordinate.
            my: Target Y coordinate.
            mz: Target Z coordinate.
            velocity: Movement velocity.
        """
        self.client.moveToPositionAsync(mx, my, mz, velocity).join()

    def _move_by_velocity(self, vx: float, vy: float, vz: float) -> None:
        """Move drone by velocity.

        Args:
            vx: X velocity component.
            vy: Y velocity component.
            vz: Z velocity component.
        """
        self.client.moveByVelocityAsync(vx, vy, vz, self.interval).join()
