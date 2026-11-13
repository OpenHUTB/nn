#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务2（神经网络版）第二步：训练"扫描 -> 速度"网络。

结构: 90 -> 96 (ReLU) -> 2     纯 NumPy，SGD + 动量
输入: ~/policy_data/scan_dataset.csv
输出: ~/policy_data/scan_policy.npz
"""

import csv
import os
import sys

import numpy as np


def load(path):
    rows = list(csv.reader(open(path)))
    head = rows[0]
    data = np.array([[float(v) for v in r] for r in rows[1:]], dtype=np.float32)
    n_in = len(head) - 2
    return data[:, :n_in], data[:, -2:]


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/policy_data")
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    x, y = load(os.path.join(root, "scan_dataset.csv"))
    print("samples:", x.shape)

    # 输入归一化 + 目标归一化
    xm, xs = x.mean(0), x.std(0) + 1e-6
    ym, ys = y.mean(0), y.std(0) + 1e-6
    xn = (x - xm) / xs
    yn = (y - ym) / ys

    # 按时间顺序切分（避免相邻帧泄漏）
    n = len(xn)
    cut = int(n * 0.85)
    xtr, ytr = xn[:cut], yn[:cut]
    xva, yva = xn[cut:], yn[cut:]

    d = x.shape[1]
    h = 96
    rng = np.random.RandomState(0)
    W1 = rng.randn(d, h).astype(np.float32) * np.sqrt(2.0 / d)
    b1 = np.zeros(h, dtype=np.float32)
    W2 = rng.randn(h, 2).astype(np.float32) * np.sqrt(2.0 / h)
    b2 = np.zeros(2, dtype=np.float32)
    mW1 = np.zeros_like(W1); mb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); mb2 = np.zeros_like(b2)

    lr, mom, l2 = 2e-3, 0.9, 1e-5
    best, best_pack = 1e9, None

    for ep in range(epochs):
        perm = np.random.permutation(len(xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb, yb = xtr[b], ytr[b]
            hh = np.maximum(xb @ W1 + b1, 0.0)
            o = hh @ W2 + b2
            e = o - yb
            go = e * (2.0 / len(b))
            gW2 = hh.T @ go + l2 * W2
            gb2 = go.sum(0)
            gh = (go @ W2.T) * (hh > 0)
            gW1 = xb.T @ gh + l2 * W1
            gb1 = gh.sum(0)
            mW1 = mom * mW1 - lr * gW1; W1 += mW1
            mb1 = mom * mb1 - lr * gb1; b1 += mb1
            mW2 = mom * mW2 - lr * gW2; W2 += mW2
            mb2 = mom * mb2 - lr * gb2; b2 += mb2

        hv = np.maximum(xva @ W1 + b1, 0.0)
        vl = float(np.mean((hv @ W2 + b2 - yva) ** 2))
        if vl < best:
            best, best_pack = vl, (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        if ep % 50 == 0 or ep == epochs - 1:
            print("epoch %3d  val %.5f" % (ep, vl))

    W1, b1, W2, b2 = best_pack
    np.savez(os.path.join(root, "scan_policy.npz"),
             W1=W1, b1=b1, W2=W2, b2=b2, xm=xm, xs=xs, ym=ym, ys=ys)
    print("best val %.5f -> scan_policy.npz" % best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
