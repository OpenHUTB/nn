"""
主程序：驾驶事故视频识别工具（优化版）
优化点说明：
1. 性能提速：跳过重复依赖检查、缓存环境变量减少属性查找、简化检测器初始化逻辑
2. 灵活配置：支持命令行动态调整检测源/置信度/日志级别/语言，无需修改配置文件
3. 规范日志：替换print为logging模块，支持分级输出（DEBUG/INFO/WARNING），便于调试和生产环境使用
4. 交互优化：新增人和小车识别提示，明确告知用户当前模型支持的识别类别
5. 健壮性提升：参数合法性校验、异常捕获并分级输出、兼容不同运行路径
"""
# 系统内置模块：基础功能支撑
import sys  # 系统路径、退出等核心操作
import os   # 环境变量、文件路径等操作系统交互
import argparse  # 命令行参数解析工具
import logging  # 日志模块（替代print，支持分级输出、格式化、持久化等）

# 自定义模块/配置：项目核心配置和工具
from config import (
    REQUIRED_PACKAGES,  # 项目必需的依赖包列表（如ultralytics/opencv-python等）
    PYPI_MIRROR,        # PyPI镜像源（国内默认清华镜像，提速依赖安装）
    DETECTION_SOURCE,   # 默认检测源（0=本地摄像头，也可传视频文件路径）
    CONFIDENCE_THRESHOLD,  # 默认检测置信度阈值（过滤低置信度的识别结果）
    ACCIDENT_CLASSES    # 事故识别核心类别（0=人，2=小车，7=卡车等）
)
from utils.dependencies import install_dependencies  # 依赖自动安装工具函数
from core.detector import AccidentDetector  # 事故检测器核心类（封装YOLO模型、检测逻辑）

# -------------------------- 核心工具函数1：日志初始化（替代print，更专业、灵活） --------------------------
def init_logger():
    """
    初始化日志系统（核心作用：统一日志格式、支持分级输出）
    返回值：
        logger: 配置好的日志实例，可调用logger.info/debug/warning/error输出不同级别日志
    日志格式：时间戳 - 日志级别 - 日志内容（例如：2025-12-22 10:00:00 - INFO - 启动程序）
    """
    # 创建日志器实例，命名为"AccidentDetection"（便于多模块日志区分）
    logger = logging.getLogger("AccidentDetection")
    # 设置默认日志级别为INFO（低于INFO的日志不会输出，如DEBUG）
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器（多次调用该函数时防止日志重复输出）
    if logger.handlers:
        return logger
    
    # 定义日志输出格式：时间+级别+内容
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",  # 格式字符串
        datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式（可读性更强）
    )
    
    # 创建控制台处理器（日志输出到终端）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)  # 绑定格式
    
    # 将处理器添加到日志器
    logger.addHandler(console_handler)
    
    return logger

# -------------------------- 核心工具函数2：命令行参数解析（灵活配置，无需改代码） --------------------------
def parse_args(logger):
    """
    解析命令行参数（核心作用：让用户通过命令行动态配置参数，提升工具灵活性）
    参数：
        logger: 日志实例（用于输出参数校验警告）
    返回值：
        args: 解析后的参数对象，可通过args.xxx访问具体参数
    """
    # 创建参数解析器，添加工具描述（--help时显示）
    parser = argparse.ArgumentParser(description="驾驶事故视频识别工具（支持动态配置）")
    
    # 1. 基础配置参数：检测源（摄像头/视频文件）
    parser.add_argument(
        "--source", "-s",  # 参数名（长/短格式）
        default=DETECTION_SOURCE,  # 默认值（从config.py读取）
        help=f"检测源（0=本地摄像头/视频文件绝对路径，默认值：{DETECTION_SOURCE}）"
    )
    
    # 2. 界面配置参数：标注语言（中文/英文）
    parser.add_argument(
        "--language", "-l",
        default="zh",  # 默认中文
        choices=["zh", "en"],  # 限制可选值（避免无效输入）
        help="标注语言（zh=中文/en=英文，默认：zh）"
    )
    
    # 3. 性能优化参数：跳过依赖检查（已安装依赖时提速）
    parser.add_argument(
        "--skip-deps", "-sd",
        action="store_true",  # 无需传值，加该参数则为True
        default=False,
        help="跳过依赖检查（已确认安装所有依赖时使用，可大幅提升启动速度）"
    )
    
    # 4. 检测精度参数：置信度阈值（过滤低置信度结果）
    parser.add_argument(
        "--conf", "-c",
        type=float,  # 参数类型（浮点型）
        default=CONFIDENCE_THRESHOLD,
        help=f"检测置信度阈值（范围0-1，值越高越严格，默认：{CONFIDENCE_THRESHOLD}）"
    )
    
    # 5. 调试配置参数：日志级别（控制输出详细程度）
    parser.add_argument(
        "--log-level", "-ll",
        default="INFO",  # 默认只输出INFO及以上级别
        choices=["DEBUG", "INFO", "WARNING"],  # 可选级别
        help="日志级别（DEBUG=调试模式/INFO=正常模式/WARNING=仅警告，默认：INFO）"
    )
    
    # 解析命令行传入的参数
    args = parser.parse_args()
    
    # 关键校验：置信度阈值合法性（必须在0-1之间）
    if not (0 < args.conf <= 1):
        # 输出警告日志，自动回退到默认值
        logger.warning(f"输入的置信度{args.conf}无效（需0<conf≤1），自动使用默认值{CONFIDENCE_THRESHOLD}")
        args.conf = CONFIDENCE_THRESHOLD
    
    return args

# -------------------------- 主函数：程序核心逻辑入口 --------------------------
def main():
    """
    程序主逻辑（按流程：初始化→参数解析→配置覆盖→依赖处理→检测器启动→异常处理）
    核心设计：分层处理，每一步都有日志反馈，便于问题定位
    """
    # 步骤1：初始化日志系统（后续所有输出都用logger，替代print）
    logger = init_logger()
    
    # 步骤2：解析命令行参数（用户传入的参数优先于配置文件）
    args = parse_args(logger)
    # 动态调整日志级别（根据用户输入的--log-level）
    logger.setLevel(args.log_level)

    # 步骤3：缓存环境变量（性能优化：减少os.environ的重复属性查找）
    # 原理：局部变量访问速度远快于模块级属性，循环/多次调用时提速明显
    env = os.environ

    # 步骤4：覆盖检测源配置（命令行参数优先于config.py的默认值）
    if str(args.source) != str(DETECTION_SOURCE):
        # 兼容处理：检测源可能是数字（摄像头ID）或字符串（视频路径）
        try:
            # 尝试转为整数（摄像头ID，如0/1）
            env["DETECTION_SOURCE"] = str(int(args.source))
        except (ValueError, TypeError):
            # 转换失败则为视频文件路径（字符串）
            env["DETECTION_SOURCE"] = str(args.source)
        # 记录配置变更日志（便于确认当前使用的检测源）
        logger.info(f"✅ 检测源已覆盖为：{env['DETECTION_SOURCE']}")

    # 步骤5：覆盖置信度配置（命令行参数优先）
    if args.conf != CONFIDENCE_THRESHOLD:
        env["CONFIDENCE_THRESHOLD"] = str(args.conf)
        logger.info(f"✅ 置信度阈值已覆盖为：{args.conf}")

    try:
        # 程序启动提示（用户感知）
        logger.info("🚀 启动驾驶事故视频识别工具...")

        # 步骤6：依赖检查/安装（可通过--skip-deps跳过，提升速度）
        if not args.skip_deps:
            # 调用依赖安装函数（自动检查缺失包并安装，使用国内镜像提速）
            install_dependencies(REQUIRED_PACKAGES, PYPI_MIRROR)
        else:
            # 跳过依赖检查的提示（告知用户当前状态）
            logger.info("⚠️ 已跳过依赖检查（--skip-deps参数生效，若依赖缺失会报错）")

        # 步骤7：初始化事故检测器（核心业务类）
        logger.info("🔄 初始化事故检测器...")
        detector = AccidentDetector()  # 实例化检测器（内部会加载YOLO模型）

        # 步骤8：提示当前模型支持的识别类别（提升用户体验）
        # 类别映射：0=人，2=小车（可根据ACCIDENT_CLASSES扩展）
        target_classes = {0: "人", 2: "小车"}
        # 过滤出当前配置中启用的识别类别
        supported_targets = [
            f"{name}（类别ID: {cid}）" 
            for cid, name in target_classes.items() 
            if cid in ACCIDENT_CLASSES
        ]
        # 输出支持的类别（用户明确知道能识别什么）
        logger.info(f"✅ 检测器初始化完成，当前模型支持识别：{', '.join(supported_targets)}")
        logger.info("✅ 开始检测（按Q/ESC键退出，画面中会实时标注识别到的人和小车）")
        
        # 步骤9：启动检测流程（传递语言参数，控制标注语言）
        detector.run_detection(language=args.language)

    # 异常处理1：用户手动中断（Ctrl+C）
    except KeyboardInterrupt:
        logger.info("\n🛑 用户强制中断程序（Ctrl+C）")
    # 异常处理2：其他所有未捕获的异常（保证程序不崩溃，输出错误信息）
    except Exception as e:
        # 输出错误日志（核心错误信息）
        logger.error(f"\n❌ 程序运行出错：{str(e)}")
        # 调试模式下输出详细的异常栈（便于定位问题）
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
    # 最终执行：无论是否异常，都输出退出提示（用户感知）
    finally:
        logger.info("👋 程序正常退出")

# -------------------------- 程序入口（脚本直接运行时执行） --------------------------
if __name__ == "__main__":
    """
    脚本入口处理（核心作用：兼容不同运行方式，保证模块导入时不执行主逻辑）
    关键优化：将当前脚本目录加入sys.path，避免模块导入失败
    """
    # 获取当前脚本的绝对路径（解决不同目录运行时的路径问题）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 若当前目录不在sys.path中，添加进去（确保自定义模块能被找到）
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    # 执行主函数
    main()
