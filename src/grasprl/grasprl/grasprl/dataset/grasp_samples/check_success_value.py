import os
import numpy as np

data_path = os.path.dirname(os.path.abspath(__file__))

success = 0
fail = 0
forces = []
sforces = []
fforces = []


files = [f for f in os.listdir(data_path) if f.endswith(".npy")]

for f in files:
    p = os.path.join(data_path, f)
    try:
        d = np.load(p, allow_pickle=True)
        if isinstance(d, np.ndarray):
            d = d.item()
        s = d.get("grasp_success", 0)
        force = d.get("grasp_force", 50)
        forces.append(force)
        if s == 1:
            success += 1
            sforces.append(force)
        else:
            fail += 1
            fforces.append(force)
    except:
        continue

print("抓取成功", success)
print("抓取失败", fail)
print("平均力", np.mean(forces) if forces else 0)
print("成功平均力", np.mean(sforces) if sforces else 0)
print("失败平均力", np.mean(fforces) if fforces else 0)