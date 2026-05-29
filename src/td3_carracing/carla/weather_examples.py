#!/usr/bin/env python3
"""
Carla 天气模式示例
这个文件展示了如何使用我们实现的多天气模式功能
"""

import sys
import os

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from carla_env import CarlaEnv, WeatherMode
from env_wrappers import wrap_env


def weather_mode_demo():
    """天气模式演示"""
    print("=== Carla 天气模式演示 ===\n")

    # 测试不同的天气模式
    weather_modes = [
        WeatherMode.CLEAR,
        WeatherMode.RAINY,
        WeatherMode.FOGGY,
        WeatherMode.CLOUDY,
        WeatherMode.WET,
        WeatherMode.RANDOM
    ]

    for weather_mode in weather_modes:
        print(f"正在测试天气模式: {weather_mode.value}")
        try:
            # 创建环境
            env = CarlaEnv(town="Town03",
                         render_mode="human",
                         max_episode_steps=100,
                         weather_mode=weather_mode)

            env = wrap_env(env)

            # 重置环境
            obs, info = env.reset()
            print(f"成功创建环境，初始观察维度: {obs.shape}")

            # 执行几个步骤
            for i in range(3):
                action = [0.0, 0.2, 0.0]  # 轻微加速
                obs, reward, terminated, truncated, info = env.step(action)

                print(f"  步骤 {i+1}:")
                print(f"    奖励: {reward:.2f}")
                print(f"    速度: {info['speed']:.1f} km/h")
                print(f"    碰撞: {info['collision']}")
                print(f"    车道入侵: {info['lane_invasion']}")

                if terminated or truncated:
                    print(f"  环境提前结束")
                    break

            env.close()
            print("-" * 50)

        except Exception as e:
            print(f"  错误: {e}")
            print("-" * 50)
            continue

    print("\n=== 天气模式演示完成 ===\n")
    print("可用的天气模式:")
    for mode in WeatherMode:
        print(f"  - {mode.value}")
    print("\n训练命令示例:")
    print("  晴天训练: python main.py --weather clear")
    print("  雨天训练: python main.py --weather rainy")
    print("  随机天气训练: python main.py --weather random")
    print("  雾天测试: python main.py --weather foggy (如果有训练好的模型会自动测试)")


def multi_weather_training():
    """多天气训练策略示例"""
    print("\n=== 多天气训练策略 ===\n")

    weather_modes = [
        WeatherMode.CLEAR,
        WeatherMode.RAINY,
        WeatherMode.FOGGY,
        WeatherMode.CLOUDY
    ]

    print("策略 1: 固定天气训练")
    print("  - 适用于特定场景的精细调优")
    print("  - 示例: 雨天训练 python main.py --weather rainy\n")

    print("策略 2: 循环天气训练")
    print("  - 轮流使用不同天气，增加模型泛化性")
    print("  - 需要修改 main.py 实现循环逻辑\n")

    print("策略 3: 随机天气训练")
    print("  - 每次重置环境时随机选择天气")
    print("  - 使用 --weather random 参数\n")

    print("策略 4: 渐进式天气训练")
    print("  - 先在简单天气(晴天)训练，再逐步增加难度")
    print("  - 需要在训练过程中调整 weather_mode 参数")


if __name__ == "__main__":
    try:
        # 运行天气模式演示
        weather_mode_demo()

        # 展示训练策略
        multi_weather_training()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序错误: {e}")
