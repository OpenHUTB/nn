import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Test start")

import os
os.makedirs('results', exist_ok=True)
print("Created results directory")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 创建一个简单的图表
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
ax.set_title('Test Plot')
plt.savefig('results/test_plot.png')
print("Saved test_plot.png")

print("Test completed successfully")