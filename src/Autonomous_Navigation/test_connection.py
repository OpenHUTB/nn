import airsim
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def connect_and_arm() -> airsim.MultirotorClient:
    """
    连接到 AirSim 模拟器，启用 API 控制并解锁无人机。

    Returns:
        已连接的客户端对象

    Raises:
        Exception: 如果连接、启用控制或解锁失败
    """
    client = airsim.MultirotorClient()
    client.confirmConnection()
    logger.info("已连接到模拟器")

    # 启用 API 控制
    if not client.enableApiControl(True):
        raise RuntimeError("启用 API 控制失败")
    logger.info("API 控制已启用")

    # 解锁无人机
    if not client.armDisarm(True):
        raise RuntimeError("解锁失败")
    logger.info("无人机已解锁")

    return client


def takeoff_and_land(client: airsim.MultirotorClient, hover_time: float = 2.0):
    """
    执行起飞、悬停和降落测试。

    Args:
        client: AirSim 客户端对象
        hover_time: 悬停时间（秒）
    """
    # 获取起飞前状态
    state = client.getMultirotorState()
    logger.info(f"起飞前状态 - 位置: {state.kinematics_estimated.position}, 速度: {state.speed:.2f} m/s")

    # 起飞
    logger.info("正在起飞...")
    client.takeoffAsync().join()
    logger.info("起飞完成，开始悬停")

    # 悬停
    time.sleep(hover_time)

    # 降落
    logger.info("正在降落...")
    client.landAsync().join()
    logger.info("降落完成")


def disarm_and_release(client: airsim.MultirotorClient):
    """锁定无人机并释放 API 控制（清理操作）"""
    try:
        client.armDisarm(False)
        client.enableApiControl(False)
        logger.info("已锁定并释放控制")
    except Exception as e:
        logger.warning(f"清理过程中出现异常: {e}")


def main():
    """主函数：连接、起飞、降落"""
    client = None
    try:
        client = connect_and_arm()
        takeoff_and_land(client, hover_time=2.0)
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.exception("执行过程中发生错误")
    finally:
        if client:
            disarm_and_release(client)
        logger.info("脚本结束")


if __name__ == "__main__":
    main()