# README
This repo is the final project of reinforcement learning course. Training a model for autonomous vehicle control in [`CARLA (version0.9.13)`](https://carla.org/). The structure of the code may be confusing and some implementations are inelegant(such as `agent.py retreive_data()`), may refactor code in the near future.      

## environment
1. carla 0.9.13 release
2. python 3.6
3. dependencies in requirements.txt

## set up env and test
Use precompiled Carla [link](https://mirrors.sustech.edu.cn/carla/carla/0.9.13/) 
```bash
# download carla from sustech mirror, you can also follow the official instruction
wget https://mirrors.sustech.edu.cn/carla/carla/0.9.13/CARLA_0.9.13.tar.gz
tar -zxvf CARLA_0.9.13.tar.gz
./CarlaUE4.sh -RenderOffScreen
# set up python environment
conda env create -f environment.yml
# test
python3 run.py or
python3 run_sac.py
```


## design details
> help to modify the algorithm  

**carla world settings:**   
using default world. Deploy and destroy the vehicles when resetting the world instead of reloading the whole world by `reload_world()`. Retrieve the rgb camera frame in synchronous mode, convert it to tensor and then store to replaybuffer.

**agent settings:**   
The agent car is always spawning  the agent at the first spawn_point with `sensor.camera.rgb` and `sensor.other.collision`.

**reward and action:**  
actions(`utility.py map2action()`):   
| action index |                  action                  |
| :----------: | :--------------------------------------: |
|      0       | go straight on(vehicle.control(1, 0, 0)) |
|      1       |   turn left(vehicle.control(1, -1, 0))   |
|      2       |   turn right(vehicle.control(1, 1, 0))   |
|      3       |     brake(vehicle.control(0, 0, 1))      |

rewards(`carlaenv.py get_reward()`)(A2C):  
| rewards |              event              |
| :-----: | :-----------------------------: |
|  -200   | collision sensor retrieve event |
|  -100   |       take action 3 brake       |
|    2    |  take action 0(go straight on)  |
|    1    |         take action 1,2         |

reward(`carlaenv.py reward_sac()`)(SAC):  
| reward |              event              |
| :----: | :-----------------------------: |
|  -200  | collision sensor retrieve event |
|   1    |             others              |


**RL algorithms:**   
currently implement A2C and SAC.


## done
- [x] wrap the environment of carla following the paradigm of OpenAI gym
  - [x] env() init the world
  - [x] step() return info
  - [x] reset() reset the world to the init status
  - [x] agent(actor)

> need to fix problem of reset environment. May using destroy() for all actors

> solution:
> use collision to indicate the episode ends.

> receive warning when destroy sensors: you should firstly sensor.stop()
> don't use reload_world(), it causes some problems(high memory usage and finally core dumped)

- [x] sample trajectories
- [x] replaybuffer
- [x] rl algorithm(actor-critic)
  - [x] generate action
  - [x] pay attention to tensor numpy conversion and detach
  - [x] need test
- [x] add SAC algorithm
- [ ] refactor code 


## notice
To run the code on my limited computation resource machine(1 rtx3060), I setting it to sample one episode and then update(online A2C). Moreover, I also directly resize and crop the frames once receiving it and store it in the replaybuffer in Tensor type to save memory. Due to the limited hardware, I just tested under a small episode length but it exactly improves.  
The reward settings can be further improved. The settings above is compared with serveral different settings. Taking brake frequently is too bad while driving. And if setting it to positive reward, the policy may learn to always brake no matter what it sees.   

**SAC**   
For SAC, the action space change to be continuous(controling steer[-1, 1]) instead of the discrete settings in A2C. Action is always in the format of carla.VehicleControl(1, steer, 0) where steer is given by the policy.

