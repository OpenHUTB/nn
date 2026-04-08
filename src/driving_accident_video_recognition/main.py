<<<<<<< HEAD
"""
主程序：驾驶事故视频识别工具（优化增强版）
新增功能：视频保存+实时统计+检测区域限定+热键扩展
"""
import sys
import os
import argparse
import logging
from config import (
    REQUIRED_PACKAGES, PYPI_MIRROR, DETECTION_SOURCE,
    CONFIDENCE_THRESHOLD, ACCIDENT_CLASSES
)
from utils.dependencies import install_dependencies
from core.detector import AccidentDetector

def init_logger():
    logger = logging.getLogger("AccidentDetection")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def parse_args(logger):
    parser = argparse.ArgumentParser(description="驾驶事故视频识别工具（支持动态配置+增强功能）")
    # 原有参数
    parser.add_argument("--source", "-s", default=DETECTION_SOURCE,
                        help=f"检测源（0=摄像头/视频路径，默认：{DETECTION_SOURCE}）")
    parser.add_argument("--language", "-l", default="zh", choices=["zh", "en"],
                        help="标注语言（zh=中文/en=英文，默认：zh）")
    parser.add_argument("--skip-deps", "-sd", action="store_true", default=False,
                        help="跳过依赖检查（已安装依赖时用，提速）")
    parser.add_argument("--conf", "-c", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"检测置信度阈值（0-1，默认：{CONFIDENCE_THRESHOLD}）")
    parser.add_argument("--log-level", "-ll", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                        help="日志级别（DEBUG=调试/INFO=正常/WARNING=仅警告，默认：INFO）")
    
    # -------------------------- 新增1：功能扩展参数 --------------------------
    # 视频保存
    parser.add_argument("--save-path", "-sp", default=None,
                        help="保存识别后视频的路径（如output.mp4，默认不保存）")
    # 实时统计
    parser.add_argument("--enable-stats", "-es", action="store_true", default=False,
                        help="启用检测统计（事故、人员、车辆数量）")
    # 检测区域限定（相对坐标x1,y1,x2,y2，范围0-1）
    parser.add_argument("--roi", "-r", type=str, default=None,
                        help="检测区域（相对坐标x1,y1,x2,y2，如0.2,0.3,0.8,0.7，默认全画面）")

    args = parser.parse_args()
    # 原有参数校验
    if not (0 < args.conf <= 1):
        logger.warning(f"置信度{args.conf}无效，自动使用默认值{CONFIDENCE_THRESHOLD}")
        args.conf = CONFIDENCE_THRESHOLD
    # 新增：检测区域参数校验
    if args.roi:
        try:
            roi_coords = list(map(float, args.roi.split(",")))
            if len(roi_coords) != 4 or not all(0 <= c <= 1 for c in roi_coords):
                raise ValueError
            args.roi = tuple(roi_coords)
            logger.info(f"检测区域已设置为：{args.roi}")
        except (ValueError, TypeError):
            logger.warning("检测区域参数无效，将使用全画面检测")
            args.roi = None
    return args

def main():
    logger = init_logger()
    args = parse_args(logger)
    logger.setLevel(args.log_level)
    env = os.environ

    # 原有：覆盖检测源和置信度
    if str(args.source) != str(DETECTION_SOURCE):
        try:
            env["DETECTION_SOURCE"] = str(int(args.source))
        except (ValueError, TypeError):
            env["DETECTION_SOURCE"] = str(args.source)
        logger.info(f"检测源已覆盖为：{env['DETECTION_SOURCE']}")
    if args.conf != CONFIDENCE_THRESHOLD:
        env["CONFIDENCE_THRESHOLD"] = str(args.conf)
        logger.info(f"置信度阈值已覆盖为：{args.conf}")

    # -------------------------- 新增2：预处理保存目录 --------------------------
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"已创建保存目录：{save_dir}")

    try:
        logger.info("🚀 启动驾驶事故视频识别工具...")
        if not args.skip_deps:
            install_dependencies(REQUIRED_PACKAGES, PYPI_MIRROR)
        else:
            logger.info("⚠️ 已跳过依赖检查（--skip-deps生效）")

        logger.info("🔄 初始化事故检测器...")
        detector = AccidentDetector()
        target_classes = {0: "人", 2: "小车"}
        supported_targets = [f"{name}（类别ID: {cid}）" for cid, name in target_classes.items() if cid in ACCIDENT_CLASSES]
        logger.info(f"✅ 检测器初始化完成，当前模型支持识别：{', '.join(supported_targets)}")
        
        # -------------------------- 新增3：提示新增热键 --------------------------
        logger.info("✅ 开始检测（热键：Q/ESC=退出，S=保存当前帧，P=暂停/继续）")
        
        # -------------------------- 新增4：传递增强参数到检测函数 --------------------------
        detector.run_detection(
            language=args.language,
            save_path=args.save_path,  # 视频保存路径
            enable_stats=args.enable_stats,  # 启用统计
            roi=args.roi  # 检测区域
        )

    except KeyboardInterrupt:
        logger.info("\n🛑 用户强制中断程序")
    except Exception as e:
        logger.error(f"\n❌ 程序运行出错：{str(e)}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
    finally:
        # -------------------------- 新增5：输出统计结果 --------------------------
        if args.enable_stats and hasattr(detector, "stats"):
            stats = detector.stats
            logger.info("\n📊 检测统计结果：")
            logger.info(f"  事故事件数：{stats.get('accident_count', 0)}")
            logger.info(f"  人员识别数：{stats.get('person_count', 0)}")
            logger.info(f"  小车识别数：{stats.get('car_count', 0)}")
        logger.info("👋 程序正常退出")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    main()
=======
"""
主程序：驾驶事故视频识别工具（优化增强版）
新增功能：视频保存+实时统计+检测区域限定+热键扩展
"""
import sys
import os
import argparse
import logging
from config import (
    REQUIRED_PACKAGES, PYPI_MIRROR, DETECTION_SOURCE,
    CONFIDENCE_THRESHOLD, ACCIDENT_CLASSES
)
from utils.dependencies import install_dependencies
from core.detector import AccidentDetector

def init_logger():
    logger = logging.getLogger("AccidentDetection")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def parse_args(logger):
    parser = argparse.ArgumentParser(description="驾驶事故视频识别工具（支持动态配置+增强功能）")
    # 原有参数
    parser.add_argument("--source", "-s", default=DETECTION_SOURCE,
                        help=f"检测源（0=摄像头/视频路径，默认：{DETECTION_SOURCE}）")
    parser.add_argument("--language", "-l", default="zh", choices=["zh", "en"],
                        help="标注语言（zh=中文/en=英文，默认：zh）")
    parser.add_argument("--skip-deps", "-sd", action="store_true", default=False,
                        help="跳过依赖检查（已安装依赖时用，提速）")
    parser.add_argument("--conf", "-c", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"检测置信度阈值（0-1，默认：{CONFIDENCE_THRESHOLD}）")
    parser.add_argument("--log-level", "-ll", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                        help="日志级别（DEBUG=调试/INFO=正常/WARNING=仅警告，默认：INFO）")
    
    # -------------------------- 新增1：功能扩展参数 --------------------------
    # 视频保存
    parser.add_argument("--save-path", "-sp", default=None,
                        help="保存识别后视频的路径（如output.mp4，默认不保存）")
    # 实时统计
    parser.add_argument("--enable-stats", "-es", action="store_true", default=False,
                        help="启用检测统计（事故、人员、车辆数量）")
    # 检测区域限定（相对坐标x1,y1,x2,y2，范围0-1）
    parser.add_argument("--roi", "-r", type=str, default=None,
                        help="检测区域（相对坐标x1,y1,x2,y2，如0.2,0.3,0.8,0.7，默认全画面）")

    args = parser.parse_args()
    # 原有参数校验
    if not (0 < args.conf <= 1):
        logger.warning(f"置信度{args.conf}无效，自动使用默认值{CONFIDENCE_THRESHOLD}")
        args.conf = CONFIDENCE_THRESHOLD
    # 新增：检测区域参数校验
    if args.roi:
        try:
            roi_coords = list(map(float, args.roi.split(",")))
            if len(roi_coords) != 4 or not all(0 <= c <= 1 for c in roi_coords):
                raise ValueError
            args.roi = tuple(roi_coords)
            logger.info(f"检测区域已设置为：{args.roi}")
        except (ValueError, TypeError):
            logger.warning("检测区域参数无效，将使用全画面检测")
            args.roi = None
    return args

def main():
    logger = init_logger()
    args = parse_args(logger)
    logger.setLevel(args.log_level)
    env = os.environ

    # 原有：覆盖检测源和置信度
    if str(args.source) != str(DETECTION_SOURCE):
        try:
            env["DETECTION_SOURCE"] = str(int(args.source))
        except (ValueError, TypeError):
            env["DETECTION_SOURCE"] = str(args.source)
        logger.info(f"检测源已覆盖为：{env['DETECTION_SOURCE']}")
    if args.conf != CONFIDENCE_THRESHOLD:
        env["CONFIDENCE_THRESHOLD"] = str(args.conf)
        logger.info(f"置信度阈值已覆盖为：{args.conf}")

    # -------------------------- 新增2：预处理保存目录 --------------------------
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"已创建保存目录：{save_dir}")

    try:
        logger.info("🚀 启动驾驶事故视频识别工具...")
        if not args.skip_deps:
            install_dependencies(REQUIRED_PACKAGES, PYPI_MIRROR)
        else:
            logger.info("⚠️ 已跳过依赖检查（--skip-deps生效）")

        logger.info("🔄 初始化事故检测器...")
        detector = AccidentDetector()
        target_classes = {0: "人", 2: "小车"}
        supported_targets = [f"{name}（类别ID: {cid}）" for cid, name in target_classes.items() if cid in ACCIDENT_CLASSES]
        logger.info(f"✅ 检测器初始化完成，当前模型支持识别：{', '.join(supported_targets)}")
        
        # -------------------------- 新增3：提示新增热键 --------------------------
        logger.info("✅ 开始检测（热键：Q/ESC=退出，S=保存当前帧，P=暂停/继续）")
        
        # -------------------------- 新增4：传递增强参数到检测函数 --------------------------
        detector.run_detection(
            language=args.language,
            save_path=args.save_path,  # 视频保存路径
            enable_stats=args.enable_stats,  # 启用统计
            roi=args.roi  # 检测区域
        )

    except KeyboardInterrupt:
        logger.info("\n🛑 用户强制中断程序")
    except Exception as e:
        logger.error(f"\n❌ 程序运行出错：{str(e)}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
    finally:
        # -------------------------- 新增5：输出统计结果 --------------------------
        if args.enable_stats and hasattr(detector, "stats"):
            stats = detector.stats
            logger.info("\n📊 检测统计结果：")
            logger.info(f"  事故事件数：{stats.get('accident_count', 0)}")
            logger.info(f"  人员识别数：{stats.get('person_count', 0)}")
            logger.info(f"  小车识别数：{stats.get('car_count', 0)}")
        logger.info("👋 程序正常退出")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    main()
>>>>>>> 377e0f2ca50fbcc5b8db8bc7644a76799e99dd1b
