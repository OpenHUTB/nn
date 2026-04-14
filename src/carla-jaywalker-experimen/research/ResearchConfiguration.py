"""添加了 __setitem__ 方法：支持字典式赋值操作
改进了 __setattr__ 方法：正确处理 _props 内部属性
添加了类型注解：使用 Dict[str, Any] 和 Any 类型
添加了文件存在性检查：使用 os.path.exists() 检查配置文件是否存在
添加了编码指定：打开文件时指定 utf-8 编码
添加了 __delitem__ 和 __delattr__ 方法：支持删除操作
添加了 __len__ 和 __iter__ 方法：支持长度获取和迭代
添加了 __repr__ 方法：提供友好的字符串表示
添加了实用方法：get(), update(), to_dict(), save()
添加了类方法 from_dict：支持从字典创建实例
改进了异常信息：提供更清晰的错误提示"""

from lib import UndefinedProperty
from typing import Dict, Any
import yaml
import os

class ResearchConfiguration:
    """
    Internal factors that affect different models
    
    This class manages configuration parameters for research models,
    supporting dictionary-style access and attribute-style access.
    """
    
    def __init__(self, defaultFactorFile: str) -> None:
        self._props = self._create_factor_dict(defaultFactorFile)
    
    def _create_factor_dict(self, defaultFactorFile: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(defaultFactorFile):
            raise FileNotFoundError(f"Configuration file not found: {defaultFactorFile}")
        
        with open(defaultFactorFile, "r", encoding='utf-8') as stream:
            config_dict = yaml.safe_load(stream)
        
        if config_dict is None:
            raise ValueError(f"No config found in {defaultFactorFile}")
        
        print("Parsed dict:", config_dict)
        return config_dict
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator"""
        return name in self._props
    
    def __getitem__(self, name: str) -> Any:
        """Support dictionary-style access: config['key']"""
        if name in self._props:
            return self._props[name]
        raise UndefinedProperty(f"ResearchConfiguration does not have '{name}'")
    
    def __setitem__(self, name: str, value: Any) -> None:
        """Support dictionary-style assignment: config['key'] = value"""
        self._props[name] = value
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Support attribute-style assignment: config.key = value"""
        if name == '_props':
            super().__setattr__(name, value)
        else:
            self._props[name] = value
    
    def __getattr__(self, name: str) -> Any:
        """Support attribute-style access: config.key"""
        if name == '_props':
            return super().__getattr__(name)
        if name in self._props:
            return self._props[name]
        raise UndefinedProperty(f"ResearchConfiguration does not have '{name}'")
    
    def __delitem__(self, name: str) -> None:
        """Support deletion: del config['key']"""
        if name in self._props:
            del self._props[name]
        else:
            raise KeyError(f"'{name}' not found")
    
    def __delattr__(self, name: str) -> None:
        """Support attribute deletion: del config.key"""
        if name in self._props:
            del self._props[name]
        else:
            raise AttributeError(f"'{name}' not found")
    
    def __len__(self) -> int:
        """Support len() function"""
        return len(self._props)
    
    def __iter__(self):
        """Support iteration"""
        return iter(self._props)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"ResearchConfiguration({self._props})"
    
    def get(self, name: str, default: Any = None) -> Any:
        """Safe get method with default value"""
        return self._props.get(name, default)
    
    def update(self, other: Dict[str, Any]) -> None:
        """Update configuration with another dictionary"""
        self._props.update(other)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return self._props.copy()
    
    def save(self, filepath: str) -> None:
        """Save current configuration to YAML file"""
        with open(filepath, 'w', encoding='utf-8') as stream:
            yaml.dump(self._props, stream, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ResearchConfiguration':
        """Alternative constructor from dictionary"""
        instance = cls.__new__(cls)
        instance._props = config_dict.copy()
        return instance

# 使用示例
# 创建配置实例
config = ResearchConfiguration("config.yaml")

# 字典式访问
value = config["some_key"]
config["new_key"] = "new_value"

# 属性式访问
value = config.some_key
config.new_key = "new_value"

# 检查是否存在
if "some_key" in config:
    print("Key exists")

# 获取长度
print(len(config))

# 迭代
for key in config:
    print(key, config[key])

# 安全获取
value = config.get("missing_key", "default_value")

# 保存配置
config.save("new_config.yaml")
