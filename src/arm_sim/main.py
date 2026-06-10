print("=== 程序启动 ===")
import mujoco
print("MuJoCo导入成功")
from simulator import IndexSimulator
print("IndexSimulator导入成功")
from task import ChoicePanelTask  # 新增：导入任务类
print("ChoicePanelTask导入成功")
import yaml  # 新增：加载配置文件
print("YAML导入成功")

def main():
    print("\n=== 进入main函数 ===")
    # 1. 配置路径
    config_path = "config.yaml"
    model_path = "simulation.xml"
    print(f"配置路径: {config_path}")
    print(f"模型路径: {model_path}")

    # 新增：加载配置文件（给任务传参）
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("[OK] 配置文件加载成功")
    except FileNotFoundError:
        print(f"[错误] 找不到配置文件 {config_path}")
        print("请检查文件是否存在，或路径是否正确")
        return
    except Exception as e:
        print(f"[错误] 配置文件加载失败: {e}")
        return

    # 2. 初始化仿真器+任务实例（带异常处理）
    try:
        sim = IndexSimulator(config_path, model_path)
        print("[OK] 仿真器初始化成功")
        task = ChoicePanelTask(config, sim)  # 关联配置和仿真器
        print("[OK] 任务模块初始化成功")
    except Exception as e:
        print(f"[错误] 仿真器/任务初始化失败: {e}")
        return

    # 3. 重置仿真和任务
    sim.reset()
    task.reset()
    print("[OK] 仿真环境重置完成，准备开始运行")

    # 4. 仿真循环（整合任务逻辑，替代sim.run_simulation()）
    try:
        # 先启动可视化viewer（和原仿真器逻辑一致）
        viewer_enabled = False
        try:
            sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            # 优化视角
            sim.viewer.cam.azimuth = 135
            sim.viewer.cam.elevation = -15
            sim.viewer.cam.distance = 0.6
            sim.viewer.cam.lookat = [0.45, -0.15, 0.8]
            viewer_enabled = True
            print("[OK] 可视化窗口启动成功！")
            print("提示：关闭窗口或按Ctrl+C可退出程序")
        except Exception as e:
            print(f"[警告] 可视化窗口启动失败: {e}")
            print("程序将以无窗口模式运行，可通过终端输出查看状态")
            sim.viewer = None

        # 循环执行仿真和任务更新
        while sim.is_running:
            # 仿真步进（模型运动）
            sim.step()
            # 任务状态更新（判断成败、计算奖励）
            task_status = task.update()

            # 如果任务完成（成功/超时），重置仿真和任务
            if task_status["done"]:
                print("\n[信息] 任务完成，准备开始新一轮...")
                sim.reset()
                task.reset()

            # 渲染同步（保持窗口流畅）
            if viewer_enabled and sim.viewer:
                try:
                    sim.viewer.sync()
                    time.sleep(0.001)
                except:
                    print("[信息] 窗口已关闭，准备退出程序...")
                    break

    except KeyboardInterrupt:
        print("\n\n[信息] 检测到Ctrl+C，正在安全退出...")
        sim.is_running = False
    finally:
        # 关闭资源
        sim.close()
        print(f"\n[OK] 仿真程序已正常结束！")
        print(f"共运行 {sim.current_step} 步")

# 新增：导入time（用于渲染延时）
import time

if __name__ == "__main__":
    main()
