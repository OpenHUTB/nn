#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务4 第二步：训练 CNN（图像 -> 控制量），行为克隆。

输入: ~/bc_data/images/*.png (64x48 灰度) + labels.csv
输出: ~/bc_data/model.pt

网络: Conv(1->8,3x3) - ReLU - MaxPool2 - Conv(8->16,3x3) - ReLU - MaxPool2
      - FC(16*12*16 -> 64) - ReLU - FC(64 -> 2)
"""

import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(16 * 12 * 16, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def load_data(root):
    import cv2
    labels = list(csv.DictReader(open(os.path.join(root, "labels.csv"))))
    xs, ys = [], []
    for r in labels:
        p = os.path.join(root, "images", r["frame"])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        xs.append(img.astype(np.float32) / 255.0)
        ys.append([float(r["v"]), float(r["w"])])
    x = torch.from_numpy(np.array(xs)[:, None, :, :])
    y = torch.from_numpy(np.array(ys, dtype=np.float32))
    return x, y


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/bc_data")
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    x, y = load_data(root)
    print("samples:", x.shape, y.shape)
    if len(x) < 20:
        print("样本太少")
        return 1

    n = len(x)
    idx = torch.randperm(n)
    split = int(n * 0.85)
    tr, va = idx[:split], idx[split:]

    net = SmallCNN()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    best = 1e9
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            out = net(x[b])
            loss = lossf(out, y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            vloss = lossf(net(x[va]), y[va]).item()
        if ep % 5 == 0 or ep == epochs - 1:
            print("epoch %3d  train %.5f  val %.5f" % (ep, tot / len(tr), vloss))
        if vloss < best:
            best = vloss
            torch.save(net.state_dict(), os.path.join(root, "model.pt"))

    print("best val loss: %.5f" % best)
    print("saved:", os.path.join(root, "model.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
