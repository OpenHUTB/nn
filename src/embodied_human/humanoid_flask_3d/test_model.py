import pyvista as pv

# 加载source文件夹里的GLB模型
model = pv.read("source\humanoid.glb")
# 初始化渲染器
plotter = pv.Plotter()
plotter.add_mesh(model, color="skyblue", edge_color="black")
plotter.add_axes()
# 显示模型
plotter.show()