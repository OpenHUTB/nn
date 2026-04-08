from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_context import LaunchContext

def generate_launch_description():
    package_dir = FindPackageShare(package='mujoco_ros_demo')
    model_path = PathJoinSubstitution([package_dir, 'config', 'humanoid.xml'])
    
    context = LaunchContext()
    resolved_model_path = model_path.perform(context)

    # 发布节点命令 - 使用 conda Python 环境
    start_pub_cmd = [
        'bash', '-c',
        f'source ~/miniconda3/bin/activate mjoco_py310 && '
        f'PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH '
        f'python -m mujoco_ros_demo.mujoco_publisher '
        f'--ros-args -r __node:=mujoco_publisher --param model_path:={resolved_model_path}'
    ]

    # 订阅节点命令 - 使用系统Python环境
    start_sub_cmd = [
        'bash', '-c',
        f'/usr/bin/python3 -m mujoco_ros_demo.data_subscriber '
        f'--ros-args -r __node:=data_subscriber'
    ]

    return LaunchDescription([
        ExecuteProcess(
            cmd=start_pub_cmd,
            output='screen'
        ),
        ExecuteProcess(
            cmd=start_sub_cmd,
            output='screen'
        )
    ])