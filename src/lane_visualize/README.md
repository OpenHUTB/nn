自动驾驶汽车车道和路径检测
=========================================
## Demo
<div align="center">
      <a href="https://www.youtube.com/watch?v=UFQQbTYH9hI-Y">
     <img 
      src="https://i.ytimg.com/vi/UFQQbTYH9hI/maxresdefault.jpg" 
      alt="Demo" 
      style="width:100%;">
      </a>
    </div>

## 如何安装

为了能够运行它，我建议使用 Python 3.6 或更高版本。

1. 安装要求

```
pip3 install -r requirements.txt
```
这将安装运行此作所需的所有依赖项。 

2. 下载示例数据

The sample data can be downloaded from [here.](https://drive.google.com/file/d/1hP-v8lLn1g1jEaJUBYJhv1mEb32hkMvG/view?usp=sharing) More data will be added soon. 

3. 运行程序

``` 
python3 main.py <path-to-sample-data-hevc> 
```

## 下一步是什么 ？ 

使用 YOLOv3 进行红绿灯、汽车、卡车、自行车、摩托车、行人和停车标志检测。
实时语义分割，或几乎实时。
快速大满贯。

## 相关研究

[Learning a driving simulator](https://arxiv.org/abs/1608.01230)

## 学分

[comma.ai for supercombo model](https://github.com/commaai/openpilot/blob/master/models/supercombo.keras)

[Harald Schafer for parts of the code](https://github.com/haraldschafer)

[lanes_image_space.py Code by @Shane](https://github.com/ShaneSmiskol)


