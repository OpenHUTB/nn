# handy description

A custom robot description for handy that includes a camera attached to the gripper.

See https://github.com/ivaROS/ivaHandy/tree/master/ros/finalarm_description for the original description.

## usage

```bash
# launch rviz
docker compose run gazebo roslaunch --wait handy_description display.launch

# launch gazebo
docker compose run gazebo roslaunch --wait handy_description gazebo.launch
```
