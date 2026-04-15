# src/box/utils/functions.py
import os
import yaml

def output_path():
    """示例：返回输出路径"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output"))

def parent_path(path):
    """示例：返回父目录路径"""
    return os.path.dirname(os.path.abspath(path))

def is_suitable_package_name(name):
    """示例：验证包名是否合法"""
    return name.isidentifier() and name[0].islower()

def parse_yaml(file_path):
    """示例：解析YAML文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(data, file_path):
    """示例：写入YAML文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)