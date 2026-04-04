import random
import numpy as np

class OU():
    """
    Ornstein-Uhlenbeck (OU) 过程类
    OU过程是一种均值回归的随机过程，常用于模拟具有均值回归特性的随机波动。
    """
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
