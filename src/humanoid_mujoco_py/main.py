# 标准库
# 导入时间模块，用于控制模拟帧率
import time

# 第三方库
# 导入MuJoCo核心库，提供物理引擎功能
import mujoco
# 从MuJoCo导入可视化工具，用于实时显示模拟过程
from mujoco import viewer

def main():
    """
    主函数：加载MuJoCo模型并运行物理模拟
    
    流程：
    1. 加载XML格式的模型文件
    2. 初始化模拟数据结构
    3. 启动可视化窗口
    4. 运行指定时长的模拟循环
    5. 输出关键模拟数据并更新可视化
    """
    # 加载MJCF模型文件
    # 模型定义了物理场景：包含平面、光源和一个自由下落的盒子
    # from_xml_path方法支持从XML文件加载模型，替代C++中的mj_loadXML
    model = mujoco.MjModel.from_xml_path("src/humanoid_control/hello.xml")
    
    # 创建与模型对应的动态数据结构
    # 存储模拟过程中的状态变量（位置、速度等），类似C++中的mjData
    data = mujoco.MjData(model)
    
    # 启动被动式可视化窗口
    # 被动模式意味着我们手动控制模拟步骤和画面更新
    # 使用with语句确保资源正确释放
    with viewer.launch_passive(model, data) as v:
        # 运行模拟（10秒，按每秒60步计算）
        # 600步 ≈ 10秒（默认时间步长为1/60秒）
        for _ in range(600):
            # 推进模拟一步
            # 该函数执行一次完整的前向动力学计算
            mujoco.mj_step(model, data)
            
            # 打印盒子的位置信息（可选）
            # data.xpos存储所有刚体的位置，是一个(n, 3)的数组
            # 索引1对应示例中的盒子（索引0是地面）
            print(f"时间: {data.time:.2f}, "
                f"位置: ({data.xpos[1, 0]:.2f}, {data.xpos[1, 1]:.2f}, {data.xpos[1, 2]:.2f})")
            
            # 更新可视化窗口
            # 将当前模拟状态同步到视图
            v.sync()
            
            # 控制可视化帧率
            # 暂停0.01秒，使可视化更流畅（实际模拟不受影响）
            time.sleep(0.01)

# 程序入口点
if __name__ == "__main__":
    # 调用主函数启动模拟
    main()
