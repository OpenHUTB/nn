import numpy as np
import os

# 你的Q表文件的准确路径（从你的截图确认）
file_path = "./agent/frozen_lake_q_table.npy"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"❌ 错误：文件不存在！")
    print(f"系统查找的路径是：{os.path.abspath(file_path)}")
    print("请确认frozen_lake_q_table.npy确实在agent/src文件夹下")
else:
    # 读取二进制Q表文件
    q_table = np.load(file_path)

    # 打印Q表基本信息
    print("=" * 60)
    print("✅ Q表读取成功！")
    print(f"Q表形状：{q_table.shape} （16个状态 × 4个动作）")
    print(f"数据类型：{q_table.dtype}")
    print("=" * 60)

    # 打印完整Q表
    print("\n📊 完整Q表内容：")
    print(q_table)

    # 打印每个状态的最优动作（数值最大的那个）
    print("\n🎯 每个状态的最优动作（0=左, 1=下, 2=右, 3=上）：")
    best_actions = np.argmax(q_table, axis=1)
    print(best_actions.reshape(4, 4))