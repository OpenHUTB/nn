#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务4：端到端神经网络（图像 -> 控制量），纯 NumPy 实现，无需 torch。

网络结构:
    输入  32x24 灰度 (768)  ->  隐层 64 (ReLU)  ->  输出 2 (v, w)
    损失  MSE + L2 正则，SGD + 动量

用法:
    python3 uav_train_np.py ~/bc_data 300
产物:
    ~/bc_data/weights.npz
"""

import csv
import os
import sys

import numpy as np


def load(root, w=32, h=24):
    import cv2
    rows = list(csv.DictReader(open(os.path.join(root, "labels.csv"))))
    xs, ys = [], []
    for r in rows:
        img = cv2.imread(os.path.join(root, "images", r["frame"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        xs.append(cv2.resize(img, (w, h)).astype(np.float32).ravel() / 255.0)
        ys.append([float(r["v"]), float(r["w"])])
    return np.array(xs), np.array(ys, dtype=np.float32)


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/bc_data")
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    x, y = load(root)
    print("samples:", x.shape, y.shape)
    if len(x) < 20:
        print("样本太少，先采集数据")
        return 1

    n, d = x.shape
    idx = np.random.permutation(n)
    split = int(n * 0.85)
    tr, va = idx[:split], idx[split:]

    # 归一化目标
    y_mean = y[tr].mean(axis=0)
    y_std = y[tr].std(axis=0) + 1e-6
    yn = (y - y_mean) / y_std

    rng = np.random.RandomState(0)
    W1 = rng.randn(d, 64).astype(np.float32) * np.sqrt(2.0 / d)
    b1 = np.zeros(64, dtype=np.float32)
    W2 = rng.randn(64, 2).astype(np.float32) * np.sqrt(2.0 / 64)
    b2 = np.zeros(2, dtype=np.float32)

    mW1 = np.zeros_like(W1); mb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); mb2 = np.zeros_like(b2)

    lr = 3e-3
    mom = 0.9
    l2 = 1e-4
    best = 1e9
    best_pack = None

    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        for i in range(0, len(perm), 64):
            b = tr[perm[i:i + 64]]
            xb, yb = x[b], yn[b]
            h = xb @ W1 + b1
            h = np.maximum(h, 0.0)
            o = h @ W2 + b2
            e = o - yb
            go = e * (2.0 / len(b))
            gW2 = h.T @ go + l2 * W2
            gb2 = go.sum(axis=0)
            gh = (go @ W2.T) * (h > 0)
            gW1 = xb.T @ gh + l2 * W1
            gb1 = gh.sum(axis=0)
            mW1 = mom * mW1 - lr * gW1; W1 += mW1
            mb1 = mom * mb1 - lr * gb1; b1 += mb1
            mW2 = mom * mW2 - lr * gW2; W2 += mW2
            mb2 = mom * mb2 - lr * gb2; b2 += mb2

        hv = np.maximum(x[va] @ W1 + b1, 0.0)
        vloss = float(np.mean((hv @ W2 + b2 - yn[va]) ** 2))
        if vloss < best:
            best = vloss
            best_pack = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        if ep % 50 == 0 or ep == epochs - 1:
            print("epoch %3d  val %.5f" % (ep, vloss))

    W1, b1, W2, b2 = best_pack
    np.savez(os.path.join(root, "weights.npz"), W1=W1, b1=b1, W2=W2, b2=b2,
             y_mean=y_mean, y_std=y_std)
    print("best val loss %.5f  ->  %s" % (best, os.path.join(root, "weights.npz")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
