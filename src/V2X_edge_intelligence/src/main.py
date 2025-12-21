#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路侧感知数据集预处理（Carla适配）
零第三方依赖（仅Python内置库），完全无报错！
运行方式：python main.py
"""
import json
import os
import random


# ===================== 第一步：生成模拟Carla数据（纯文本，不用图片） =====================
def generate_demo_data():
    """生成模拟Carla标注数据（纯文本，无需图片/OpenCV，零报错）"""
    os.makedirs("demo_carla_data", exist_ok=True)
    # 生成Carla场景标注（纯JSON文本，模拟感知数据）
    anno_data = {
        "carla_scenes": [
            {"scene_id": 1001, "frame_id": 0,
             "obstacles": [{"type": "car", "bbox": [100, 100, 200, 200], "distance": 8.5}]},
            {"scene_id": 1002, "frame_id": 1,
             "obstacles": [{"type": "person", "bbox": [150, 150, 250, 250], "distance": 5.2}]}
        ]
    }
    with open("demo_carla_data/carla_anno.json", "w", encoding="utf-8") as f:
        json.dump(anno_data, f, indent=2)
    print("✅ 模拟Carla标注数据生成完成 → demo_carla_data/carla_anno.json")


# ===================== 第二步：数据增强（文本层面模拟，无需图片） =====================
def simple_augment():
    """模拟数据增强（文本层面扩充，比如添加噪声、复制数据）"""
    with open("demo_carla_data/carla_anno.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 模拟增强：为每个场景添加随机噪声（模拟图像增强）
    augmented_data = []
    for scene in data["carla_scenes"]:
        # 复制场景并添加噪声
        aug_scene = scene.copy()
        aug_scene["aug_type"] = "random_brightness"  # 模拟亮度增强
        # 给障碍物距离加随机噪声
        for obs in aug_scene["obstacles"]:
            obs["distance"] = round(obs["distance"] + random.uniform(-0.5, 0.5), 2)
        augmented_data.append(aug_scene)

    # 保存增强后数据
    with open("demo_carla_data/carla_anno_augmented.json", "w", encoding="utf-8") as f:
        json.dump({"carla_scenes_augmented": augmented_data}, f, indent=2)
    print("✅ 数据增强完成 → demo_carla_data/carla_anno_augmented.json")


# ===================== 第三步：数据集划分（纯文本，内置库实现） =====================
def split_dataset():
    """划分数据集（8:1:1，纯文本处理）"""
    with open("demo_carla_data/carla_anno.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data["carla_scenes"]
    random.shuffle(scenes)  # 随机打乱
    total = len(scenes)
    train_size = int(total * 0.8)
    val_size = int((total - train_size) / 2)

    # 划分数据
    train_data = scenes[:train_size]
    val_data = scenes[train_size:train_size + val_size]
    test_data = scenes[train_size + val_size:]

    # 保存划分结果
    split_result = {
        "train_scenes": train_data,
        "val_scenes": val_data,
        "test_scenes": test_data,
        "split_ratio": "train:80% | val:10% | test:10%"
    }
    with open("demo_carla_data/carla_split_result.json", "w", encoding="utf-8") as f:
        json.dump(split_result, f, indent=2)
    print("✅ 数据集划分完成 → demo_carla_data/carla_split_result.json")
    print(f"   划分结果：训练集{len(train_data)}条 | 验证集{len(val_data)}条 | 测试集{len(test_data)}条")


# ===================== 主函数：一键运行 =====================
if __name__ == "__main__":
    print("===== 路侧感知数据集预处理（Carla适配） =====\n")
    generate_demo_data()
    simple_augment()
    split_dataset()
    print("\n🎉 所有步骤运行完成！生成文件列表：")
    for file in os.listdir("demo_carla_data"):
        print(f"  - demo_carla_data/{file}")