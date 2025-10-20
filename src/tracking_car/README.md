```markdown
# 基于CARLA的智能代理系统

## 项目概述

本项目旨在利用神经网络技术，实现CARLA模拟器中车辆和行人的全栈智能代理，涵盖感知、规划与控制三大核心模块。通过深度学习算法赋予虚拟智能体环境理解、决策规划和动态控制能力，同时包含具身人仿真、机械臂控制等扩展功能，构建多智能体协同的仿真系统。

## 环境配置

* **支持平台**：Windows 10/11，Ubuntu 20.04/22.04
* **核心软件**：
  * Python 3.7-3.12（需兼容3.7版本）
  * PyTorch（优先采用，不依赖TensorFlow）
  * CARLA 0.9.11+（推荐0.9.13/0.9.15版本）
* **依赖安装**：
  ```bash
  # 基础依赖
  pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
  
  # CARLA客户端安装（需替换为对应版本）
  pip install carla==0.9.15
  
  # 文档生成工具
  pip install mkdocs
  ```

## 文档生成

1. 安装文档工具链：
   ```bash
   pip install mkdocs -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
   pip install -r requirements.txt
   ```

2. 构建并预览文档：
   ```bash
   # 进入项目根目录
   cd nn
   
   # 构建静态文档
   mkdocs build
   
   # 启动本地文档服务
   mkdocs serve
   ```

3. 浏览器访问 [http://127.0.0.1:8000](http://127.0.0.1:8000) 查看文档。

## 核心功能模块

1. **车辆智能代理**
   * **感知系统**：基于CNN的目标检测（车辆、行人、交通信号灯）、车道线识别
   * **规划系统**：全局路径规划（A*、RRT*算法）、局部避障
   * **控制系统**：强化学习车辆控制（油门、刹车、转向）、PID参数自适应调优
   * **手动控制**：支持键盘操作的车辆交互模式（WASD控制方向，空格/左Shift控制升降）

2. **具身人仿真**
   * **感知模块**：william开发的具身人环境感知系统（`humanoid_perception`）
   * **运动模拟**：基于Mujoco的物理引擎实现具身人运动控制（`Mujoco_manrun`）

3. **神经网络模型**
   * **CNN模型**：图像识别与目标检测（`chap05_CNN`），包含完整训练流程（Adam优化器、交叉熵损失）
   * **RNN模型**：序列数据处理与生成（`chap06_RNN`），基于LSTM实现唐诗生成等文本任务
   * **FNN模型**：基础神经网络结构（`chap04_simple_neural_network`），包含测试评估函数

4. **辅助系统**
   * **车道辅助**：Active-Lane-Keeping-Assistant实现车道偏离预警与辅助
   * **机械臂测试**：humantest模块提供机械臂力控仿真与交互功能

## 快速启动

1. 启动CARLA服务器：
   ```bash
   # Linux
   ./CarlaUE4.sh
   
   # Windows
   CarlaUE4.exe
   ```

2. 运行示例场景：
   ```bash
   # 自动驾驶车辆（强化学习）
   python src/driverless_car/main.py
   
   # 车辆手动控制
   python src/manual_control/main.py
   
   # CNN模型训练
   python src/chap05_CNN/CNN_pytorch.py
   
   # RNN文本生成
   python src/chap06_RNN/tangshi_for_pytorch/rnn.py
   ```

## 关键代码说明

### 神经网络训练框架
```python
# CNN训练示例（chap05_CNN/CNN_pytorch.py）
def train(cnn):
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(max_epoch):
        for step, (x_, y_) in enumerate(train_loader):
            x, y = Variable(x_), Variable(y_)
            output = cnn(x)
            loss = loss_func(output, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if step != 0 and step % 20 == 0:
                print(f"测试准确率: {test(cnn)}")
```

### 强化学习智能体
```python
# 车辆强化学习（driverless_car/main.py）
def main():
    env = DroneEnv()
    agent = Agent()
    episodes = 1000
    epsilon = 1.0  # 探索率
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.get_action(state, epsilon)  # 根据状态和探索率获取动作
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)  # 存储经验
            agent.train(batch_size=32)  # 训练模型
            
            if done:
                epsilon = max(0.01, epsilon * 0.995)  # 衰减探索率
                break
```

### RNN模型结构
```python
# 唐诗生成RNN（chap06_RNN/tangshi_for_pytorch/rnn.py）
class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()
        self.word_embedding_lookup = word_embedding
        self.rnn_lstm = nn.LSTM(input_size=embedding_dim, 
                                hidden_size=lstm_hidden_dim, 
                                num_layers=2, 
                                batch_first=False)
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, sentence, is_test=False):
        batch_input = self.word_embedding_lookup(sentence).view(1, -1, self.word_embedding_dim)
        h0 = torch.zeros(2, batch_input.size(1), self.lstm_dim)
        c0 = torch.zeros(2, batch_input.size(1), self.lstm_dim)
        
        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))
        out = self.fc(output.contiguous().view(-1, self.lstm_dim))
        return self.softmax(out)
```

## 贡献指南

请在提交代码前阅读 [贡献指南](https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md)，代码优化方向包括：

* 遵循 [PEP 8 代码风格](https://peps.pythonlang.cn/pep-0008/) 并完善注释
* 实现神经网络在CARLA场景中的端到端应用
* 撰写模块功能说明与API文档
* 添加自动化测试（模型性能、场景稳定性、数据一致性）
* 优化感知-规划-控制链路的实时性

## 参考资源

* [CARLA官方文档](https://carla.readthedocs.io/)
* [PyTorch神经网络教程](https://pytorch.org/tutorials/)
* [项目文档中心](https://openhutb.github.io/nn/)
* [代理模拟器文档](https://openhutb.github.io/carla_doc/)
* [神经网络原理](https://github.com/OpenHUTB/neuro)