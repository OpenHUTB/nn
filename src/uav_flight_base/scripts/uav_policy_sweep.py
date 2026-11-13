# -*- coding: utf-8 -*-
"""网络结构 / 优化器对比实验（作业要求：同算法不同模型结构的对比）。

数据: policy_data/scan_dataset.csv （90 维激光扫描 -> (v, w)）
对比: 隐层宽度 32 / 64 / 96 / 128 / 256 × {SGD+Momentum, Adam}
产出: policy_data/sweep_results.csv
"""
import csv
import os
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "policy_data", "scan_dataset.csv")
OUT = os.path.join(BASE, "policy_data", "sweep_results.csv")

EPOCHS = 300
BATCH = 64
L2 = 1e-5
SEED = 0


def load():
    df = pd.read_csv(CSV)
    x = df.iloc[:, :-2].to_numpy(dtype=np.float32)
    y = df.iloc[:, -2:].to_numpy(dtype=np.float32)
    return x, y


def train(x, y, hidden, opt_name):
    rng = np.random.RandomState(SEED)
    n, d = x.shape
    cut = int(n * 0.85)
    xtr, ytr, xva, yva = x[:cut], y[:cut], x[cut:], y[cut:]

    xm, xs = xtr.mean(0), xtr.std(0) + 1e-6
    ym, ys = ytr.mean(0), ytr.std(0) + 1e-6
    xtr = (xtr - xm) / xs
    xva = (xva - xm) / xs
    ytr = (ytr - ym) / ys
    yva = (yva - ym) / ys

    W1 = rng.randn(d, hidden).astype(np.float32) * np.sqrt(2.0 / d)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.randn(hidden, 2).astype(np.float32) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(2, dtype=np.float32)

    lr = 2e-3 if opt_name == "sgd" else 1e-3
    mom, b1m, b2m = 0.9, 0.9, 0.999
    mW1 = np.zeros_like(W1); mb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); mb2 = np.zeros_like(b2)
    vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
    t = 0

    best = 1e9
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(xtr))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb, yb = xtr[b], ytr[b]
            h = np.maximum(xb @ W1 + b1, 0.0)
            o = h @ W2 + b2
            go = (o - yb) * (2.0 / len(b))
            gW2 = h.T @ go + L2 * W2
            gb2 = go.sum(0)
            gh = (go @ W2.T) * (h > 0)
            gW1 = xb.T @ gh + L2 * W1
            gb1 = gh.sum(0)

            t += 1
            if opt_name == "sgd":
                mW1 = mom * mW1 - lr * gW1; W1 += mW1
                mb1 = mom * mb1 - lr * gb1; b1 += mb1
                mW2 = mom * mW2 - lr * gW2; W2 += mW2
                mb2 = mom * mb2 - lr * gb2; b2 += mb2
            else:
                for P, G, M, V in ((W1, gW1, mW1, vW1), (b1, gb1, mb1, vb1),
                                   (W2, gW2, mW2, vW2), (b2, gb2, mb2, vb2)):
                    M *= b1m
                    M += (1 - b1m) * G
                    V *= b2m
                    V += (1 - b2m) * G * G
                    Mh = M / (1 - b1m ** t)
                    Vh = V / (1 - b2m ** t)
                    P -= lr * Mh / (np.sqrt(Vh) + 1e-8)

        hv = np.maximum(xva @ W1 + b1, 0.0)
        vl = float(np.mean((hv @ W2 + b2 - yva) ** 2))
        if vl < best:
            best = vl

    return {
        "hidden": hidden,
        "optimizer": opt_name,
        "params": d * hidden + hidden + hidden * 2 + 2,
        "val_loss": round(best, 5),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    x, y = load()
    print("samples:", x.shape)
    rows = []
    for hidden in (32, 64, 96, 128, 256):
        for opt in ("sgd", "adam"):
            r = train(x, y, hidden, opt)
            rows.append(r)
            print(r)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["hidden", "optimizer", "params", "val_loss", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
