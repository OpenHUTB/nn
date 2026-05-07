"""
根据 https://microsoft.github.io/AirSim/voxel_grid/ 生成世界地图
"""

import os
import airsim

c = airsim.VehicleClient()
c.confirmConnection()

center = airsim.Vector3r(0, 0, 0)
output_path = os.path.join(os.getcwd(), "map.binvox")

ok = c.simCreateVoxelGrid(center, 200, 200, 100, 0.5, output_path)
print(f"simCreateVoxelGrid returned: {ok}")
print(f"output path: {output_path}")
print(f"file exists: {os.path.exists(output_path)}")
