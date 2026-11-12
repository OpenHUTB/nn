from setuptools import setup

package_name = "car_nn_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/car_nn.launch.py", "launch/car_slam.launch.py", "launch/car_teleop.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zhoul",
    maintainer_email="zhoul@example.com",
    description="MuJoCo car + neural network driving demo",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_node = car_nn_sim.sim_node:main",
            "slam_node = car_nn_sim.slam_node:main",
        ],
    },
)
