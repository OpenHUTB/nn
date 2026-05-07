# Working Space

After creating a ros working space, replace the src folder with the src folder here. Note that you may re-run ```catkin_make``` command to make sure caches being created properly.
Please install the required packages listed in *requirements.txt* first.
Please follow the instructions [here](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros1/) to download ```ros-bridge``` first. I used its source repo (i.e. method B).
The working space directory should be placed under *PythonAPI* directory(i.e. same folder as *examples* is).

## packages

### ros-bridge
This folder contains default ros functionalities provided by official tutorial. There are lots of awesome tools you would be interested in. Of course you have to download carla first.

### main
Note that ```collect.launch``` will execute some nodes here. ```img_subs.py``` and ```pcl_subs.py``` are responsilbe for storing images and point cloud data respectively. **You must repalce FOLDER with the storage location you prefer.** these data comes from the sensor which is simulated by carla. also, you can do something on these data(e.g. add 3d bounding box) ```main.py``` publishes the image data, in this case you can use RVIZ to view the result repeatedly. ```show_prediction_2d.py``` should draw the bounding box, but I've not finished it yet so currently this .py doesn't work at all. this .py is copied from the AVOD project. check AVOD to see more details.

## launch files

```main.launch``` will create a simulation space with an ego vehicle and some sensors on it. Practically, I just trigger some build-in packages together to achieve this. The official website introduces these packages completely so let me skip this part. ```collect.launch``` will trigger ```img_subs.py``` and ```pcl_subs.py``` at the same time, so you don't have to execute them one-by-one.

## How to Use
After execute ```CarlaUE4.sh```, execute ```roslaunch main main``` to initialize ego vehicle; execute ```roslaunch main collect``` to collect PCL and image received from the sensor.

## Reference
1. [ROS Bridge](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros1/)
2. [ROS Launch](https://ithelp.ithome.com.tw/articles/10209542)
