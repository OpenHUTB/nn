mkdir -p src/mobl_pointing_159588/assets

cat > src/mobl_pointing_159588/main.py << 'PY'
#!/usr/bin/env python3
"""
Minimal demo runner for 'mobl_arms_index_pointing' with uitb.
- Runs a short episode without rendering (headless-friendly).
- If uitb is not installed, prints guidance and exits gracefully.
"""
import sys

def run():
    try:
        from uitb import Simulator
    except Exception as e:
        print("未检测到 uitb（user-in-the-box）。请先在目标机器按其 README 安装依赖后再运行。")
        print("原始错误：", e)
        sys.exit(0)

    sim = Simulator.get("simulators/mobl_arms_index_pointing")
    obs, info = sim.reset()
    for _ in range(200):
        obs, r, term, trunc, info = sim.step(sim.action_space.sample())
        if term or trunc:
            break
    sim.close()
    print("✓ 完成一次无渲染运行。录制视频示例命令见本目录 README。")

if __name__ == "__main__":
    run()
PY

chmod +x src/mobl_pointing_159588/main.py
