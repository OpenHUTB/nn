自动驾驶车道与路径检测
=========================================
## 在 GTAV 上运行 OpenPilot
本项目是对 littlemountainman/modeld项目的一个分支。
我们利用了他的工作，并将 DeepGTAV 和 VPilot 结合，从而能够将 comma.ai 的开源软件应用于 GTAV，并创建出由 openpilot 算法管理的自动驾驶车辆。

## 如何安装

为了能够运行此项目，我推荐使用 Python 3.7 或更高版本。

1. 安装所需依赖包

```
pip3 install -r requirements.txt
```

这将安装运行此项目所需的所有必要依赖。

2. 下载 [Vpilot with DeepGTAV](https://github.com/aitorzip/VPilot)

3. 下载 [ScriptHook](https://www.dev-c.com/gtav/scripthookv/)

4. 下载 [DeepGTA](https://github.com/aitorzip/DeepGTAV)

5. 将 ScriptHookV.dll、dinput8.dll、NativeTrainer.asi文件复制到游戏的主文件夹，即 GTA5.exe所在的目录。

6. 将 DeepGTAV/bin/Release文件夹下的所有内容复制并粘贴到你的 GTAV 游戏安装目录下。

7. 启动程序（确保 GTAV 已在运行）
```
python3 main.py
```


## 感谢

[littlemountainman/modeld](https://github.com/littlemountainman/modeld)
[aitorzip/DeepGTAV](https://github.com/aitorzip/DeepGTAV)
[aitorzip/VPilot](https://github.com/aitorzip/VPilot)

