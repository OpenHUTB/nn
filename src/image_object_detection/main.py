# 导入核心库
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os  # 新增：用于路径验证

# -------------------------- 1. 基础配置（重点：替换成你的图片路径！） --------------------------
# 模型路径：YOLOv8n轻量级预训练模型（自动下载）
MODEL_PATH = "yolov8n.pt"

# 🔥 关键修改：替换成你图片的绝对路径（右键图片→属性→复制完整路径，加r前缀避免转义）
# 示例：IMAGE_PATH = r"C:\Users\apple\OneDrive\桌面\my_test_image.jpg"
IMAGE_PATH = r"C:\Users\apple\OneDrive\桌面\test.jpg"  

# 检测结果保存路径（建议保存到桌面，方便查找）
SAVE_PATH = r"C:\Users\apple\OneDrive\桌面\detected_image.jpg"

# -------------------------- 2. 加载YOLO模型 --------------------------
# 加载预训练YOLOv8模型（首次运行自动下载权重，已下载则直接加载）
model = YOLO(MODEL_PATH)

# -------------------------- 3. 图像检测核心函数（含路径验证） --------------------------
def detect_image_with_pretrained_model(image_path, save_path):
    """
    用预训练YOLO模型检测图像，包含路径验证和友好报错
    :param image_path: 待检测图片路径
    :param save_path: 检测结果保存路径
    """
    # 第一步：验证图片路径是否存在（核心解决FileNotFoundError）
    if not os.path.exists(image_path):
        print(f"\n❌ 错误：找不到图片文件！")
        print(f"当前设置的图片路径：{image_path}")
        print(f"请检查：1. 图片是否存在 2. 路径是否正确 3. 路径无中文/空格/特殊符号\n")
        return  # 路径错误则终止函数
    
    # 第二步：执行目标检测（conf=0.25：只显示置信度≥25%的目标）
    print(f"\n✅ 开始检测图片：{image_path}")
    results = model(image_path, conf=0.25)
    
    # 第三步：可视化检测结果（绘制边界框、类别、置信度）
    annotated_image = results[0].plot()  # 生成带标注的图片
    
    # 转换颜色通道（OpenCV默认BGR，Matplotlib显示需要RGB）
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # 第四步：显示检测结果图片
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image_rgb)
    plt.axis("off")  # 隐藏坐标轴
    plt.title("YOLOv8 Object Detection Result", fontsize=16)
    plt.show()
    
    # 第五步：保存检测结果到指定路径
    cv2.imwrite(save_path, annotated_image)
    print(f"\n✅ 检测结果已保存：{save_path}")
    
    # 第六步：打印详细检测信息（类别、置信度、坐标）
    print("\n📌 检测到的目标信息：")
    for result in results:
        boxes = result.boxes  # 获取所有检测框
        if len(boxes) == 0:
            print("   未检测到任何目标（可降低conf阈值试试，比如conf=0.1）")
            continue
        for box in boxes:
            cls_index = int(box.cls)  # 类别索引
            cls_name = model.names[cls_index]  # 类别名称（如person/car/cat）
            confidence = box.conf.item()  # 置信度
            coordinates = box.xyxy.tolist()[0]  # 边界框坐标 [x1, y1, x2, y2]
            print(f"   类别：{cls_name} | 置信度：{confidence:.2f} | 坐标：{[round(x, 2) for x in coordinates]}")

# -------------------------- 4. 自定义数据集训练函数（可选） --------------------------
def train_custom_yolo_model(data_yaml_path, epochs=10, imgsz=640):
    """
    训练自定义YOLO模型（需先准备数据集和.yaml配置文件）
    :param data_yaml_path: 数据集配置文件路径（如dataset/data.yaml）
    :param epochs: 训练轮数（入门建议10-30）
    :param imgsz: 输入图像尺寸
    """
    if not os.path.exists(data_yaml_path):
        print(f"\n❌ 错误：数据集配置文件不存在！路径：{data_yaml_path}")
        return
    
    # 加载模型并开始训练
    train_model = YOLO(MODEL_PATH)
    train_results = train_model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,  # 自动适配批次大小
        device="cpu",  # 无GPU则用cpu，有GPU改0
        patience=50,
        save=True,
        project="runs/train",
        name="custom_yolo",
        exist_ok=True
    )
    # 验证模型
    val_results = train_model.val()
    print("\n✅ 自定义模型训练完成！验证集指标：", val_results.results_dict)

# -------------------------- 主程序运行入口 --------------------------
if __name__ == "__main__":
    # 运行预训练模型检测（核心功能，必执行）
    detect_image_with_pretrained_model(IMAGE_PATH, SAVE_PATH)
    
    # 如需训练自定义数据集，取消下面注释并配置data_yaml_path
    # train_custom_yolo_model(data_yaml_path=r"C:\Users\apple\OneDrive\桌面\dataset\data.yaml", epochs=10)



