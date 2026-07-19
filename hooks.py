"""
MkDocs hooks 模块
自动将文档版权信息中的年份替换为当前年份
"""
from datetime import datetime


def on_config(config, **kwargs):
    """
    MkDocs 配置钩子函数，在构建时自动更新版权年份。

    Args:
        config: MkDocs 配置对象，包含 copyright 字段
        **kwargs: 其他关键字参数

    Returns:
        无返回值，直接修改 config.copyright 字段
    """
    year = str(datetime.now().year)
    config.copyright = config.copyright.format(year=year)
