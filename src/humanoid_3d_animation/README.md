# 具身人3D可视化与动作交互网页开发
基于Python+Flask+PyVista的3D可视化系统，支持模型视角控制、颜色切换、基础动作交互，通过GitHub Pages实现公开访问。

## 项目概述
本项目以Flask为Web框架，结合PyVista/Trimesh实现具身人3D模型的Web化渲染，完成基础交互（视角旋转/缩放、颜色切换）与进阶动作控制（抬手、抬腿等预设动作），并通过GitHub完成代码管理与静态网页部署，最终实现可公开访问的具身人3D交互网页。

## 环境准备
### 开发工具
- PyCharm（社区版）：项目管理、编码与调试
- GitHub：代码托管与Pages部署
- Chrome/Firefox：Web功能测试

### 依赖库安装
```bash
# 虚拟环境下执行
pip install flask pyvista trimesh numpy
```
| 依赖库       | 功能说明                     |
|-------------|------------------------------|
| flask       | Web框架，实现路由与页面渲染   |
| pyvista     | 3D可视化核心，支持模型Web适配 |
| trimesh     | 模型关节控制，处理带骨骼模型  |
| numpy       | 数值计算，实现关节角度转换    |

### 模型准备
下载带骨骼绑定的具身人3D模型（GLB/FBX格式），推荐来源：
- Sketchfab官网搜索「Free Rigged Humanoid GLB」
- 筛选条件：Downloadable + Low Poly（＜50MB）
- 模型放置路径：项目根目录（命名为`humanoid_rigged.glb`）

## 项目结构
| 文件名/目录               | 功能描述                                                     |
|--------------------------|--------------------------------------------------------------|
| `app.py`                 | Flask核心入口，实现首页路由与Web服务启动                     |
| `test_model.py`          | 本地模型加载与关节控制测试脚本                               |
| `generate_web_model.py`  | 将本地3D模型转换为Web可加载的JS文件                         |
| `generate_animations.py` | 生成预设动作（初始/抬手/抬腿）的Web模型文件                  |
| `build_static.py`        | 将动态项目转换为静态网页文件，适配GitHub Pages部署            |
| `templates/index.html`   | 网页模板，包含UI布局、交互按钮、3D渲染容器                   |
| `static/js/3d_render.js` | 3D渲染核心脚本，实现模型加载、颜色切换、动作交互             |
| `static/js/*.js`         | 自动生成的模型Web文件（基础模型+动作模型）                   |
| `static_site/`           | 静态网页输出目录，用于GitHub Pages部署                       |
| `README.md`              | 项目说明文档                                                 |

## 核心功能
### 1. 基础3D可视化（Web端）
- 模型渲染：加载带关节的具身人3D模型，支持Web端流畅显示
- 视角交互：鼠标拖拽旋转模型、滚轮缩放视角
- 颜色切换：一键切换模型颜色（蓝色/绿色/粉色）

### 2. 动作交互控制
- 预设动作：支持初始姿态、抬手、抬腿3种基础动作切换
- 关节控制：基于Trimesh修改模型关节角度，实现精准动作控制
- 动作适配：所有动作模型预生成Web文件，保证前端加载效率

### 3. 工程化部署
- 代码管理：通过GitHub实现版本控制，支持代码同步与回溯
- 静态部署：生成纯静态网页文件，通过GitHub Pages实现公开访问

## 使用方法
### 本地开发与测试
#### 步骤1：环境与模型验证
```bash
# 测试本地模型加载
python test_model.py
```
验证效果：弹出3D窗口显示模型，终端输出关节列表。

#### 步骤2：生成Web模型文件
```bash
# 生成基础模型Web文件
python generate_web_model.py
# 生成动作模型Web文件
python generate_animations.py
```
生成结果：`static/js/`目录下新增`humanoid_base.js`、`anim_*.js`文件。

#### 步骤3：启动本地Web服务
```bash
python app.py
```
访问地址：`http://127.0.0.1:5000`
验证功能：
- 模型正常渲染，支持视角交互
- 颜色按钮切换模型颜色
- 动作按钮切换初始/抬手/抬腿姿态

### GitHub部署（公开访问）
#### 步骤1：生成静态文件
```bash
python build_static.py
```
生成结果：项目根目录新增`static_site/`，包含完整静态网页文件。

#### 步骤2：GitHub Pages部署
1. 打开项目GitHub仓库 → Settings → Pages
2. 选择「Upload your files」，上传`static_site/`内所有文件
3. 等待5-10分钟，访问仓库Pages链接即可公开访问

## 交互操作指南
| 操作类型       | 操作方式                          | 效果说明                     |
|----------------|-----------------------------------|------------------------------|
| 视角控制       | 鼠标拖拽模型区域                  | 360°旋转模型视角             |
| 缩放控制       | 滚轮滚动模型区域                  | 放大/缩小模型显示比例        |
| 颜色切换       | 点击「蓝色/绿色/粉色」按钮        | 实时切换模型主体颜色         |
| 动作切换       | 点击「初始姿态/抬手/抬腿」按钮    | 切换模型至对应预设动作姿态   |

## 常见问题与解决方案
| 问题现象               | 原因分析                          | 解决方法                                                      |
|------------------------|-----------------------------------|---------------------------------------------------------------|
| 模型加载失败           | 模型格式错误/路径错误             | 确认模型为GLB/FBX带关节格式，检查代码中模型文件名是否匹配      |
| 动作切换无效果         | 关节名称不匹配                    | 运行`test_model.py`查看实际关节名，修改`generate_animations.py`中关节配置 |
| GitHub Pages访问异常   | 文件路径错误/部署未完成           | 检查`static_site/`文件完整性，等待部署完成后清除浏览器缓存重试 |
| Web渲染卡顿            | 模型面数过多                      | 更换Low Poly模型，或用Blender简化模型面数                     |


## 参考资料
- [Flask官方文档](https://flask.palletsprojects.com/)
- [PyVista Web可视化文档](https://docs.pyvista.org/guide/web/index.html)
- [Trimesh模型操作文档](https://trimsh.org/)
- [GitHub Pages部署指南](https://docs.github.com/zh/pages)
- [Sketchfab免费3D模型资源](https://sketchfab.com/)