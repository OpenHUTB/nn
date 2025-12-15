这是manayume的使用深度学习的自动驾驶汽车导航项目测试
参考项目：https://github.com/varunpratap222/Autonomous-Vehicle-Navigation-Using-Deep-Learning.git
测试环境：使用ubuntu20.04版本，conda虚拟环境，python3.7版本，安装的软件包按照requirements.txt文件安装，使用Carla0.9.13版本进行模拟


启动流程
## Run
1. Run Carla Server using: `./CarlaUE4.sh`
2. Run `config.py` file to load Town02
3. Generate traffic either using `generate_traffic.py` or spawn pedestrians at random location along Trajectory 1 and 2 using the `pedestrians_1.py` and `pedestrians_2.py` script. Change the number of vehicles and pedestrians to spawn using `generate_traffic.py` by passing the corresponding arguments.
4. Select an existing trajectory (Trajectory 1, Trajectory 2, Trajectory 3, Trajectory 4) or set custom trajectory using the format given in `test_everything.py` arguments.
5. To find the initial and final locations of the custom trajectory, make use of `get_location.py` file and navigate the map using W,A,S,D, E, and Q keys. 
6. Enter locations or select existing trajectories and run the `test_everything.py` file.




