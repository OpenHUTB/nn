"""
create_config_files.py - 创建完整的实验配置文件（最终版本）
"""

from pathlib import Path

# 完整的配置模板
BASE_CONFIG = '''# {title}

GYM_ENV:
  # 基础参数
  TARGET_SPEED: 30.0
  MAX_SPEED: 40.0
  MAX_ACC: 5.0
  TRACK_LENGTH: 1000
  LOOK_BACK: 20
  TIME_STEP: 10
  LOOP_BREAK: 5
  DISTN_FRM_VHCL_AHD: 40
  FIXED_REPRESENTATION: true

  # 观测配置
  USE_LOCAL_OBS: true
  SENSOR_RANGE: 50.0
  USE_COOP_PERCEPTION: false
  COOP_RANGE: 150.0
  USE_PREDICTION: false
  PREDICTION_HORIZON: 3.0
  PREDICTION_DT: 0.5

  # 安全层
  USE_SAFETY_LAYER: true
  MIN_SAFE_DIST: 10.0
  MIN_TTC: 3.0

  # Lyapunov参数
  LYAPUNOV_OMEGA_N: 2.0
  LYAPUNOV_ZETA: 0.7
  LYAPUNOV_TTC_MIN: 2.5
  LYAPUNOV_D_MIN: 5.0
  LYAPUNOV_D_COMFORTABLE: 20.0
  LYAPUNOV_ALPHA_LATERAL: 0.5
  LYAPUNOV_ALPHA_LONGITUDINAL: 0.3
  LYAPUNOV_SIGMA_SAFETY: 5.0
  LYAPUNOV_WEIGHT: {lyapunov_weight}
  DISABLE_SIMPLE_LC_SHAPING: {disable_simple_lc}

POLICY:
  NAME: 'PPO2'
  NET: 'LSTM'
  CNN_EXTRACTOR: 'nature_cnn'
  ACTION_NOISE: 0.1
  PARAM_NOISE_STD: 0.2

CARLA:
  LANE_WIDTH: 3.5
  DT: 0.05
  MAX_S: 2000

TRAFFIC_MANAGER:
  N_SPAWN_CARS: 50
  MIN_SPEED: 20.0
  MAX_SPEED: 40.0
  PERC_CARS_OBEY_TRAFFIC_LIGHTS: 0.8
  PERC_IGNORE_TRAFFIC_LIGHTS: 0.2
  GLOBAL_DISTANCE_TO_LEADING_VEHICLE: 2.5
  VEHICLE_LANE_OFFSET: 0.0
  GLOBAL_PERCENTAGE_SPEED_DIFFERENCE: 30.0

LOCAL_PLANNER:
  MIN_SPEED: 0.0
  MAX_SPEED: 40.0
  MAX_ACCEL: 5.0
  MAX_CURVATURE: 1.0
  MAX_ROAD_WIDTH_L: 7.0
  MAX_ROAD_WIDTH_R: 7.0
  D_ROAD_W: 0.5
  DT: 0.2
  MAXT: 5.0
  MINT: 4.0
  D_T_S: 0.5
  N_S_SAMPLE: 2
  TARGET_SPEED: 30.0
  D_D: 0.5
  KJ: 0.1
  KT: 0.1
  KD: 1.0
  KLAT: 1.0
  KLON: 1.0
'''

CONFIGS = {
    'experiment_baseline.yaml': {
        'title': 'Baseline实验配置（无换道塑形）',
        'lyapunov_weight': '0.0',
        'disable_simple_lc': 'true'
    },
    'experiment_improved.yaml': {
        'title': 'Improved实验配置（简单换道塑形）',
        'lyapunov_weight': '0.0',
        'disable_simple_lc': 'false'
    },
    'experiment_lyapunov.yaml': {
        'title': 'Lyapunov实验配置（理论驱动塑形）',
        'lyapunov_weight': '0.30',
        'disable_simple_lc': 'true'
    }
}


def create_config_files():
    """创建配置文件"""
    config_dir = Path('tools/cfgs')
    config_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Creating COMPLETE experiment configuration files...")
    print("=" * 80 + "\n")

    for filename, params in CONFIGS.items():
        file_path = config_dir / filename

        # 生成配置内容
        content = BASE_CONFIG.format(**params)

        # 检查是否已存在
        if file_path.exists():
            backup_path = file_path.with_suffix('.yaml.backup')
            import shutil
            shutil.copy2(file_path, backup_path)
            print("[BACKUP] {} -> {}".format(file_path, backup_path))

        # 写入新文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print("[CREATED] {}".format(file_path))
        print("  Title: {}".format(params['title']))
        print("  Lyapunov Weight: {}".format(params['lyapunov_weight']))
        print("  Disable Simple LC: {}".format(params['disable_simple_lc']))
        print()

    print("=" * 80)
    print("[OK] All configuration files created!")
    print("=" * 80)

    print("\n[INFO] Configuration sections added:")
    print("  ✓ GYM_ENV (20+ parameters)")
    print("  ✓ POLICY")
    print("  ✓ CARLA")
    print("  ✓ TRAFFIC_MANAGER (8 parameters)")
    print("  ✓ LOCAL_PLANNER (15 parameters)")  # 新增

    print("\nFiles created:")
    for filename in CONFIGS.keys():
        print("  - tools/cfgs/{}".format(filename))

    print("\nYou can now run:")
    print("  python experiments/scripts/run_experiments.py --mode quick_test")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    create_config_files()