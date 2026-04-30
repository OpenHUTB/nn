#!/usr/bin/env python
# coding: utf-8

# ======================================================
# 实验名称：RNN学习大整数加法的进位机制
# ======================================================
# 思路：
#   1. 随机生成两个整数作为加法的输入，计算它们的和。
#   2. 将整数拆分成数位（低位在前，高位在后），方便RNN逐位学习“进位”规律。
#   3. 构建RNN模型，输入是两个数的数位序列，输出是逐位的预测和。
#   4. 训练模型，让它学会模拟加法。
#
#   📌 RNN的优势在于：低位的结果会影响高位（进位），这种时序依赖非常适合RNN建模。
# ======================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import os
import json
from pathlib import Path

# 运行参数（支持环境变量覆盖）
RANDOM_SEED = int(os.getenv("LEARN2CARRY_SEED", "42"))
TRAIN_STEPS = int(os.getenv("LEARN2CARRY_TRAIN_STEPS", "3000"))
TRAIN_BATCH = int(os.getenv("LEARN2CARRY_TRAIN_BATCH", "200"))
EVAL_BATCH = int(os.getenv("LEARN2CARRY_EVAL_BATCH", "2000"))
MAXLEN = int(os.getenv("LEARN2CARRY_MAXLEN", "11"))
LOG_INTERVAL = int(os.getenv("LEARN2CARRY_LOG_INTERVAL", "50"))
REPORT_OUT = os.getenv("LEARN2CARRY_REPORT_OUT", "outputs/learn2carry_report.json")

# 固定随机种子，保证结果可复现
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ======================================================
# 一、数据处理函数
# ======================================================

def gen_data_batch(batch_size: int, start: int, end: int) -> tuple:
    """
    随机生成一批加法数据

    Args:
        batch_size: 批量大小
        start: 随机数范围下限（包含）
        end: 随机数范围上限（不包含）

    Returns:
        (nums1, nums2, results)
            nums1: 第一个加数数组
            nums2: 第二个加数数组
            results: 两数之和数组
    """
    nums1 = np.random.randint(start, end, batch_size)
    nums2 = np.random.randint(start, end, batch_size)
    results = nums1 + nums2
    return nums1, nums2, results


def num_to_digits(num: int) -> list:
    """整数 → 数位列表，例如 133412 -> [1, 3, 3, 4, 1, 2]"""
    return [int(ch) for ch in str(num)]


def digits_to_num(digits: list) -> int:
    """数位列表 → 整数，例如 [1, 2, 3] -> 123"""
    return int("".join(map(str, digits)))


def pad_digits(digits: list, length: int, pad: int = 0) -> list:
    """填充数位列表到固定长度（右边补pad），例如 [1,2] -> [1,2,0,0]"""
    return digits + [pad] * (length - len(digits))


def batch_prepare(nums1, nums2, results, maxlen: int):
    """
    批量数据预处理：
        1. 转换为数位
        2. 翻转数位（低位在前，高位在后，符合加法规则）
        3. 填充到固定长度

    Returns:
        nums1_digits, nums2_digits, results_digits
    """
    def process_num(n):  # 提取转换函数：数字→数位→反转→填充
        return pad_digits(num_to_digits(n)[::-1], maxlen)
    nums1_digits = [process_num(n) for n in nums1]
    nums2_digits = [process_num(n) for n in nums2]
    results_digits = [process_num(r) for r in results]
    return nums1_digits, nums2_digits, results_digits


def digits_batch_to_numlist(batch_digits: list) -> list:
    """批量将预测的数位列表还原为整数"""
    return [digits_to_num(list(reversed(d))) for d in batch_digits]


# ======================================================
# 二、模型定义
# ======================================================

class RNNAdder(keras.Model):
    """RNN大数加法模型"""

    def __init__(self):
        super().__init__()
        # 嵌入层：数字 0~9 -> 32维向量
        self.embed = layers.Embedding(input_dim=10, output_dim=32)

        # RNN层：学习进位机制
        self.rnn = layers.RNN(layers.SimpleRNNCell(64), return_sequences=True)

        # 输出层：预测每个位上的数字（0-9）
        self.dense = layers.Dense(10)

    @tf.function
    def call(self, num1, num2):
        """
        前向传播
        Args:
            num1: [batch, maxlen] 第一个加数
            num2: [batch, maxlen] 第二个加数
        Returns:
            logits: [batch, maxlen, 10] 每个位的预测概率分布
        """
        # 嵌入
        emb1 = self.embed(num1)  # [B, L, 32]
        emb2 = self.embed(num2)  # [B, L, 32]

        # 拼接输入
        x = tf.concat([emb1, emb2], axis=-1)  # [B, L, 64]

        # RNN输出
        rnn_out = self.rnn(x)  # [B, L, 64]

        # 每个位的预测
        logits = self.dense(rnn_out)  # [B, L, 10]
        return logits


# ======================================================
# 三、训练与评估
# ======================================================

@tf.function
def compute_loss(logits, labels):
    """交叉熵损失"""
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(losses)


@tf.function
def train_step(model, optimizer, num1, num2, labels):
    """单步训练"""
    with tf.GradientTape() as tape:
        logits = model(num1, num2)
        loss = compute_loss(logits, labels)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(model, optimizer, steps=1000, batch_size=200, log_interval=50, maxlen=11):
    """训练过程"""
    final_loss = None
    for step in range(steps):
        # 生成训练数据
        data = gen_data_batch(batch_size, 0, 555_555_555)
        nums1, nums2, results = batch_prepare(*data, maxlen=maxlen)

        # 单步训练
        loss = train_step(model, optimizer,
                          tf.constant(nums1, dtype=tf.int32),
                          tf.constant(nums2, dtype=tf.int32),
                          tf.constant(results, dtype=tf.int32))
        final_loss = float(loss.numpy())

        if step == 0 or (step + 1) % log_interval == 0:
            print(f"Step {step + 1:04d}: Loss = {loss.numpy():.4f}")
    return final_loss


def evaluate(model, batch_size=2000, maxlen=11, preview_count=20):
    """评估模型精度"""
    # 生成测试数据（更大范围）
    data = gen_data_batch(batch_size, 555_555_555, 999_999_999)
    nums1, nums2, results = batch_prepare(*data, maxlen=maxlen)

    # 预测
    logits = model(tf.constant(nums1, dtype=tf.int32),
                   tf.constant(nums2, dtype=tf.int32))
    preds = np.argmax(logits.numpy(), axis=-1)

    # 转换为整数
    pred_nums = digits_batch_to_numlist(preds)

    # 打印部分预测
    for truth, pred in list(zip(data[2], pred_nums))[:preview_count]:
        print(f"真实值: {truth:<12} 预测值: {pred:<12} 正确吗: {truth == pred}")

    # 计算准确率
    acc = np.mean([t == p for t, p in zip(data[2], pred_nums)])
    print(f"\n整体准确率: {acc:.4f}")
    return acc


# ======================================================
# 四、主程序入口
# ======================================================

if __name__ == "__main__":
    if TRAIN_STEPS <= 0 or TRAIN_BATCH <= 0 or EVAL_BATCH <= 0 or LOG_INTERVAL <= 0 or MAXLEN <= 0:
        raise ValueError("配置参数必须为正整数（steps/batch/log_interval/maxlen）")

    model = RNNAdder()
    optimizer = optimizers.Adam(0.001)

    print("开始训练...")
    final_loss = train(
        model,
        optimizer,
        steps=TRAIN_STEPS,
        batch_size=TRAIN_BATCH,
        log_interval=LOG_INTERVAL,
        maxlen=MAXLEN,
    )

    print("\n模型评估：")
    acc = evaluate(model, batch_size=EVAL_BATCH, maxlen=MAXLEN)

    # 保存实验报告（便于提交与复现）
    report_path = Path(REPORT_OUT)
    if not report_path.is_absolute():
        report_path = Path(__file__).resolve().parent / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": RANDOM_SEED,
        "train_steps": TRAIN_STEPS,
        "train_batch": TRAIN_BATCH,
        "eval_batch": EVAL_BATCH,
        "maxlen": MAXLEN,
        "log_interval": LOG_INTERVAL,
        "final_loss": final_loss,
        "accuracy": float(acc),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("评估报告已保存:", report_path.resolve())
