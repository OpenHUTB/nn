from easydict import EasyDict
from pathlib import Path
import yaml


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, cfg):
    """
    从 YAML 文件读取配置到 cfg（EasyDict）
    统一使用 UTF-8 编码 + yaml.FullLoader，避免 Windows 下 GBK / Loader 报错
    """
    # 用 UTF-8 打开，避免 gbk 解码失败
    with open(cfg_file, 'r', encoding='utf-8') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)  # 或 yaml.safe_load(f)

    # 如果 cfg 是 EasyDict，通常这里用 update 就行
    try:
        cfg.update(new_config)
    except AttributeError:
        # 有些版本是用 __dict__ 或直接赋值的，你也可以根据自己原始代码改成原来的操作
        for k, v in new_config.items():
            setattr(cfg, k, v)

    return cfg


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0

# ===== 初始化 RL 配置 =====
cfg.RL = EasyDict()

# ===== 在config.py文件末尾添加 =====

# Lyapunov理论塑形配置
LYAPUNOV_OMEGA_N = 2.0           # 自然频率 (rad/s)
LYAPUNOV_ZETA = 0.7              # 阻尼比
LYAPUNOV_TTC_MIN = 2.5           # 最小TTC (s)
LYAPUNOV_D_MIN = 5.0             # 最小安全距离 (m)
LYAPUNOV_D_COMFORTABLE = 20.0    # 舒适距离 (m)
LYAPUNOV_ALPHA_LATERAL = 0.5     # 侧向势能权重
LYAPUNOV_ALPHA_LONGITUDINAL = 0.3  # 纵向势能权重
LYAPUNOV_SIGMA_SAFETY = 5.0      # 安全势能陡峭度

# Lyapunov塑形总权重
LYAPUNOV_WEIGHT = 0.30           # Lyapunov塑形在总奖励中的权重

# 其他配置
DISABLE_SIMPLE_LC_SHAPING = True  # 禁用简单换道塑形（避免与Lyapunov冲突）

# 在配置文件中添加（如果没有）
cfg.RL.W_SPEED = 10
cfg.RL.W_R_SPEED = 10
cfg.RL.MIN_SPEED_GAIN = 0.05
cfg.RL.MIN_SPEED_LOSS = -0.05
cfg.RL.LANE_CHANGE_REWARD = 1.2
cfg.RL.LANE_CHANGE_PENALTY = 0.3
cfg.RL.COLLISION = -100
cfg.RL.OFF_THE_ROAD = -50