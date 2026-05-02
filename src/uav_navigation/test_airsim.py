import airsim
import sys


def print_environment_info():
    """
    Print Python and AirSim version information.
    """
    print("Python 版本:", sys.version)

    try:
        print("AirSim 版本:", airsim.__version__)
        print("AirSim 导入成功！")
    except AttributeError:
        print("无法获取 AirSim 版本信息")


def create_client_safe():
    """
    Safely create an AirSim client.

    Returns:
        client or None
    """
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("客户端创建成功")
        return client
    except Exception as e:
        print("客户端创建失败:", e)
        return None


def main():
    """
    Main test entry.
    """
    print_environment_info()

    client = create_client_safe()

    if client is None:
        print("AirSim 未运行，跳过后续测试")
    else:
        print("AirSim 连接正常")


if __name__ == "__main__":
    main()