"""
主程序：驾驶事故视频识别工具（优化版）
优化点：性能提速+灵活配置+规范日志+新增人和小车识别提示
"""
import sys
import os
import argparse
import logging  # 新增：日志模块（替代print，支持分级输出）
from config import (
    REQUIRED_PACKAGES, PYPI_MIRROR, DETECTION_SOURCE,
    CONFIDENCE_THRESHOLD, ACCIDENT_CLASSES  # 新增：引入识别类别配置
)
from utils.dependencies import install_dependencies
from core.detector import AccidentDetector

# 在 main.py 的 init_logger 函数中添加一行（关闭日志传播，避免重复输出）
def init_logger():
    logger = logging.getLogger("AccidentDetection")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 新增：避免日志被父logger重复输出
    # 控制台输出格式：时间+日志级别+内容
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
# -------------------------- 新增2：优化命令行参数（更灵活的配置） --------------------------
def parse_args(logger):
    parser = argparse.ArgumentParser(description="驾驶事故视频识别工具（支持动态配置）")
    # 基础参数：检测源、语言
    parser.add_argument("--source", "-s", default=DETECTION_SOURCE,
                        help=f"检测源（0=摄像头/视频路径，默认：{DETECTION_SOURCE}）")
    parser.add_argument("--language", "-l", default="zh", choices=["zh", "en"],
                        help="标注语言（zh=中文/en=英文，默认：zh）")
    # 新增：性能/配置参数（无需改config.py，直接命令行调整）
    parser.add_argument("--skip-deps", "-sd", action="store_true", default=False,
                        help="跳过依赖检查（已安装依赖时用，提速）")
    parser.add_argument("--conf", "-c", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"检测置信度阈值（0-1，默认：{CONFIDENCE_THRESHOLD}）")
    # 新增：日志级别（调试/正常模式切换）
    parser.add_argument("--log-level", "-ll", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                        help="日志级别（DEBUG=调试/INFO=正常/WARNING=仅警告，默认：INFO）")
    
    args = parser.parse_args()
    # 校验参数合法性（新增：避免无效输入）
    if not (0 < args.conf <= 1):
        logger.warning(f"置信度{args.conf}无效，自动使用默认值{CONFIDENCE_THRESHOLD}")
        args.conf = CONFIDENCE_THRESHOLD
    return args

# -------------------------- 优化3：主函数逻辑（减少重复计算+提升健壮性+新增人和小车识别） --------------------------
def main():
    # 初始化日志
    logger = init_logger()
    # 解析参数（并应用日志级别）
    args = parse_args(logger)
    logger.setLevel(args.log_level)  # 动态调整日志级别

    # -------------------------- 优化4：缓存环境变量操作（减少属性查找，提速） --------------------------
    env = os.environ  # 局部变量缓存os.environ，避免循环中重复查找（参考摘要5“缓存属性”）
    # 覆盖检测源（命令行优先）
    if str(args.source) != str(DETECTION_SOURCE):
        # 严谨处理检测源类型：尝试转整数（摄像头），失败则为字符串（视频路径）
        try:
            env["DETECTION_SOURCE"] = str(int(args.source))  # 摄像头（数字）
        except (ValueError, TypeError):
            env["DETECTION_SOURCE"] = str(args.source)  # 视频路径（字符串）
        logger.info(f"检测源已覆盖为：{env['DETECTION_SOURCE']}")

    # 覆盖置信度阈值（命令行优先）
    if args.conf != CONFIDENCE_THRESHOLD:
        env["CONFIDENCE_THRESHOLD"] = str(args.conf)
        logger.info(f"置信度阈值已覆盖为：{args.conf}")

    try:
        logger.info("🚀 启动驾驶事故视频识别工具...")
        # -------------------------- 优化5：跳过依赖检查（避免重复安装，提速） --------------------------
        if not args.skip_deps:
            install_dependencies(REQUIRED_PACKAGES, PYPI_MIRROR)
        else:
            logger.info("⚠️ 已跳过依赖检查（--skip-deps生效）")

        # -------------------------- 优化6：简化检测器初始化（减少冗余代码） --------------------------
        logger.info("🔄 初始化事故检测器...")
        detector = AccidentDetector()
        # 新增：提示当前模型支持识别人和小车
        target_classes = {0: "人", 2: "小车"}
        supported_targets = [f"{name}（类别ID: {cid}）" for cid, name in target_classes.items() if cid in ACCIDENT_CLASSES]
        logger.info(f"✅ 检测器初始化完成，当前模型支持识别：{', '.join(supported_targets)}")
        logger.info("✅ 开始检测（按Q/ESC退出，画面中会标注识别到的人和小车）")
        
        # 启动检测（传递语言参数）
        detector.run_detection(language=args.language)

    except KeyboardInterrupt:
        logger.info("\n🛑 用户强制中断程序")
    except Exception as e:
        # 新增：DEBUG级别输出详细异常栈，INFO级别只显示错误信息（方便调试）
        logger.error(f"\n❌ 程序运行出错：{str(e)}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
    finally:
        logger.info("👋 程序正常退出")

if __name__ == "__main__":
    # 新增：确保code目录在搜索路径（兼容不同运行方式）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    main()
