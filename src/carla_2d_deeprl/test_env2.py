"""
test_env.py — 第6次提交完整测试脚本
===============================
测试覆盖：
  1) 独立配置文件导入验证
  2) 不同观测尺寸（动态修改）
  3) 天气预设切换
  4) 地图切换
  5) 渲染/fast/debug 开关分离
  6) 奖励系数配置验证
  7) 自定义 reward_config 传入
  8) 环境 reset/step/close 完整流程

运行方式：
  python test_env.py

前提条件：
  - Carla 服务器正在运行（localhost:2000）
  - 已安装依赖：pip install -r requirements.txt
"""

import carla
import sys
import copy
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 测试 0：配置导入验证
# ============================================================


def test_config_import():
    """验证所有配置项可从 config.py 正常导入"""
    from min_carla_env.config import (
        CONFIG,
        REWARD_CONFIG,
        WEATHER_PRESETS,
        MAP_OPTIONS,
        ACTIONS,
    )
    assert "width" in CONFIG, "CONFIG 缺少 width"
    assert "height" in CONFIG, "CONFIG 缺少 height"
    assert "max_step" in CONFIG, "CONFIG 缺少 max_step"
    assert "render" in CONFIG, "CONFIG 缺少 render"
    assert len(ACTIONS) == 3, "ACTIONS 应有 3 个动作"
    assert "clear_noon" in WEATHER_PRESETS, "WEATHER_PRESETS 缺少 clear_noon"
    assert "town02" in MAP_OPTIONS, "MAP_OPTIONS 缺少 town02"
    assert "lane_center_reward" in REWARD_CONFIG, "REWARD_CONFIG 缺少 lane_center_reward"
    assert "speed_reward_scale" in REWARD_CONFIG, "REWARD_CONFIG 缺少 speed_reward_scale"
    assert "stuck_penalty" in REWARD_CONFIG, "REWARD_CONFIG 缺少 stuck_penalty"
    logger.info("✅ 测试 0 通过：所有配置项导入正常")


# ============================================================
# 测试 1：不同观测尺寸
# ============================================================
def test_obs_size(client):
    """测试不同观测尺寸动态配置"""
    from min_carla_env.env import CarlaEnv, CONFIG

    test_sizes = [
        {"width": 240, "height": 240, "name": "小尺寸 240x240"},
        {"width": 480, "height": 480, "name": "默认 480x480"},
    ]

    for size in test_sizes:
        cfg = copy.deepcopy(CONFIG)
        cfg["width"] = size["width"]
        cfg["height"] = size["height"]

        env = CarlaEnv(client, cfg, world_config={
            "render": False,
            "fast": True,
            "town": "Town02"
        }, debug=False)

        obs = env.reset()
        expected_shape = (size["height"], size["width"])
        assert obs.shape == expected_shape, \
            f"{size['name']}: 期望 {expected_shape}, 实际 {obs.shape}"

        env.close()
        logger.info(f"✅ 观测尺寸 {size['name']} 正常")


# ============================================================
# 测试 2：天气预设切换
# ============================================================
def test_weather_presets(client):
    """测试不同天气预设"""
    from min_carla_env.env import CarlaEnv, CONFIG
    from min_carla_env.config import WEATHER_PRESETS

    weather_tests = ["clear_noon", "wet_noon", "clear_sunset"]

    for wname in weather_tests:
        env = CarlaEnv(client, CONFIG, world_config={
            "render": False,
            "fast": True,
            "town": "Town02",
            "weather": WEATHER_PRESETS[wname]
        }, debug=False)

        # 验证天气设置生效
        env.reset()
        world_weather = env.mw.world.get_weather()
        logger.info(f"  天气 {wname}: sun_altitude={world_weather.sun_altitude_angle:.1f}")
        env.close()
        logger.info(f"✅ 天气 {wname} 设置正常")


# ============================================================
# 测试 3：地图切换
# ============================================================
def test_map_switch(client):
    """测试运行时地图切换"""
    from min_carla_env.env import CarlaEnv, CONFIG
    from min_carla_env.config import MAP_OPTIONS

    env = CarlaEnv(client, CONFIG, world_config={
        "render": False,
        "fast": True,
        "town": MAP_OPTIONS["town02"]
    }, debug=False)

    obs = env.reset()
    assert obs is not None, "Town02 初始化失败"
    logger.info("✅ Town02 初始化成功")

    # 切换到 Town07
    env.mw.change_map(MAP_OPTIONS["town07"])
    obs = env.reset()
    assert obs is not None, "Town07 切换失败"
    logger.info("✅ 地图切换到 Town07 成功")

    env.close()


# ============================================================
# 测试 4：render / fast / debug 开关分离
# ============================================================
def test_switches(client):
    """测试 render、fast、debug 三个开关分离控制"""
    from min_carla_env.env import CarlaEnv, CONFIG

    # 4a) render=False + fast=True + debug=False（训练模式）
    cfg_train = copy.deepcopy(CONFIG)
    cfg_train["render"] = False
    cfg_train["fast"] = True
    env_train = CarlaEnv(client, cfg_train, world_config={
        "render": False, "fast": True, "town": "Town02"
    }, debug=False)
    obs = env_train.reset()
    assert obs is not None
    logger.info("✅ 训练模式 (render=False, fast=True, debug=False) 正常")
    env_train.close()

    # 4b) render=True + fast=False + debug=True（调试模式）
    cfg_debug = copy.deepcopy(CONFIG)
    cfg_debug["render"] = True
    env_debug = CarlaEnv(client, cfg_debug, world_config={
        "render": True, "fast": False, "town": "Town02"
    }, debug=True)
    obs = env_debug.reset()
    assert obs is not None
    logger.info("✅ 调试模式 (render=True, fast=False, debug=True) 正常")
    env_debug.close()

    # 4c) render=False + fast=False + debug=False（离线基准）
    cfg_offline = copy.deepcopy(CONFIG)
    cfg_offline["render"] = False
    env_offline = CarlaEnv(client, cfg_offline, world_config={
        "render": False, "fast": False, "town": "Town02"
    }, debug=False)
    obs = env_offline.reset()
    assert obs is not None
    logger.info("✅ 离线模式 (render=False, fast=False, debug=False) 正常")
    env_offline.close()


# ============================================================
# 测试 5：奖励系数配置 + 自定义 reward_config
# ============================================================
def test_reward_config(client):
    """测试奖励系数可配置性和自定义 reward_config"""
    from min_carla_env.env import CarlaEnv, CONFIG
    from min_carla_env.config import REWARD_CONFIG

    # 5a) 默认奖励配置
    env_default = CarlaEnv(client, CONFIG, world_config={
        "render": False, "fast": True, "town": "Town02"
    }, debug=False)
    assert env_default.reward_config == REWARD_CONFIG
    logger.info("✅ 默认奖励配置加载正常")

    # 5b) 自定义奖励配置
    custom_reward = copy.deepcopy(REWARD_CONFIG)
    custom_reward["lane_center_reward"] = 1.0   # 加强车道中心奖励
    custom_reward["speed_reward_scale"] = 0.05   # 提高速度奖励
    custom_reward["stuck_penalty"] = 200.0        # 加重卡住惩罚

    env_custom = CarlaEnv(client, CONFIG, reward_config=custom_reward, world_config={
        "render": False, "fast": True, "town": "Town02"
    }, debug=False)
    assert env_custom.reward_config["lane_center_reward"] == 1.0
    assert env_custom.reward_config["stuck_penalty"] == 200.0
    logger.info("✅ 自定义奖励配置传入正常")

    # 清理
    env_default.close()
    env_custom.close()


# ============================================================
# 测试 6：reset / step / close 完整流程
# ============================================================
def test_full_workflow(client):
    """测试完整的环境交互流程"""
    from min_carla_env.env import CarlaEnv, CONFIG

    env = CarlaEnv(client, CONFIG, world_config={
        "render": False, "fast": True, "town": "Town02"
    }, debug=False)

    # 重置
    obs = env.reset()
    assert obs is not None, "Reset 返回 None"

    # 执行 20 步
    total_reward = 0.0
    for step_idx in range(20):
        action = step_idx % 3  # 轮换 0/1/2 动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        logger.debug(f"  Step {step_idx + 1}: action={action}, reward={reward:.4f}, done={done}")
        if done:
            logger.info(f"  第 {step_idx + 1} 步结束 (reward={total_reward:.2f})")
            break

    logger.info(f"  total_reward={total_reward:.2f}")
    logger.info("✅ 完整流程 (reset → step×20 → close) 正常")

    env.close()


# ============================================================
# 主测试入口
# ============================================================
def run_all_tests():
    print("=" * 60)
    print("  min-carla-env 第6次提交 · 完整测试")
    print("=" * 60)
    print()

    # 连接 Carla（一次性复用）
    logger.info("连接 Carla 服务器 (localhost:2000)...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    try:
        client.get_server_version()
        logger.info("✅ Carla 连接成功")
    except Exception as e:
        logger.error(f"❌ Carla 连接失败: {e}")
        logger.error("请确保 Carla 服务器正在运行")
        sys.exit(1)

    tests = [
        ("0. 配置导入验证", test_config_import),
        ("1. 观测尺寸动态修改", lambda: test_obs_size(client)),
        ("2. 天气预设切换", lambda: test_weather_presets(client)),
        ("3. 地图切换", lambda: test_map_switch(client)),
        ("4. render/fast/debug 开关分离", lambda: test_switches(client)),
        ("5. 奖励系数 + 自定义 reward_config", lambda: test_reward_config(client)),
        ("6. 完整 reset/step/close 流程", lambda: test_full_workflow(client)),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\n{'─' * 40}")
            print(f"  测试 {name}")
            print(f"{'─' * 40}")
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"❌ 测试 {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"  测试完成: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
