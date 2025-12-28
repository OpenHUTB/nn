1.训练完好的权重模型会保存到saved_mode里面，eval板块会直接调用dqn的权重模型进行评估，我训练了三个小时的模型权重已经放在里面了，可以直接使用
下载链接通过网盘分享的文件：dqn权重模型
链接: https://pan.baidu.com/s/1Pdorjhu2OaBUCKC1o3xHtw?pwd=2vib 提取码: 2vib 
通过网盘分享的文件：DoubleDQN.pt
链接: https://pan.baidu.com/s/1Xo6fd0mccjkjFX99zLWhWA?pwd=4u6y 提取码: 4u6y 
使用方法，把权重模型文件放在training目录下面的saved_models目录下，eval即可自己调用

2.代码路径需要改成本地的，需要注意

3.下图是运行入口训练代码三个小时的奖励曲线图

![rewardline](https://raw.githubusercontent.com/GODDDDD22311/assets/main/rewardline.PNG)

4.configs里的文件是训练模型的超参数配置文件，可以改参数

5.注意需要安装cu121(cuda)依赖，即需要启动gpu进行训练，cpu训练可能需要以天为单位进行训练，可以新建终端用gpucheck.txt里的指令查看显存+利用率
下图为指令实图

![gpucheck](https://raw.githubusercontent.com/GODDDDD22311/assets/main/gpucheck.gif)


6.plot代码使用教程，mac用户需要注意你们的日志生成和windows用户不一样，windows用户行与行有空格，mac用户没有空格，图片生成时可以把下面的代码模块改一下
源代码

win版
with open(log_path, 'r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader if row]   # 关键：跳过空行
    reward_row = [float(x) for x in rows[2][1:] if x]


mac版
 with open(log_path, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

7.文件目录创建教程
日志生成在Log里，模型权重放在saved_mode里，这些东西全都在training目录里，录视频的videos里是dqn和ddqn

