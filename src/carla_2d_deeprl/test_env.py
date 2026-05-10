import carla
import time
from min_carla_env.env import CarlaEnv, CONFIG


def test_carla_env():
    # 1. 连接Carla客户端
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)

    # 2. 初始化环境
    try:
        env = CarlaEnv(client, CONFIG, world_config={
            "render": True,
            "fast": True,
            "town": "Town02"
        }, debug=True)
        print("环境初始化成功")

        # 3. 测试reset
        obs = env.reset()
        print(f"Reset成功，观测维度: {obs.shape if obs is not None else 'None'}")

        # 4. 测试step（执行100步）
        total_reward = 0
        for step in range(100):
            action = 0  # 直行
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print(f"Step {step + 1}: 奖励={reward:.2f}, 累计奖励={total_reward:.2f}, Done={done}")
            time.sleep(0.1)
            if done:
                print("提前结束，重置环境")
                env.reset()

        # 5. 测试地图切换
        env.mw.change_map("Town07")
        print("地图切换到Town07成功")
        env.reset()

    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        # 6. 关闭环境
        if 'env' in locals():
            env.close()
        print("环境已关闭")


if __name__ == "__main__":
    test_carla_env()