# RadarFields-with-data

基于[雷达场](https://github.com/princeton-computational-imaging/RadarFields)进行修改，提供了数据

## 与源代码的不同之处

- 增加了数据梳理脚本makedata.py
- 修改了渲染函数
- 每个batch获得数据后在进行随机采样
- 修改损失函数来获得更强的二值化

## 安装

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install ninja
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## 数据准备

将雷达图像数据放入data/data/radar目录之下

```bash
python data/data/makedataV2.py
```

这里使用的数据是牛津雷达数据集[The Oxford Radar RobotCar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets/2019-01-10-11-46-21-radar-oxford-10k)，该数据集可以兼容

也可以是用我提供的demo数据集 具体数据在example_data目录之下

## 运行模型

```bash
python main.py --config configs/radarfields.ini
```

## 注意

- 请尽可能使用radarfields.ini下的相关配置，删除分之后可能存在bug，例如如果不使用tinycudann可能会存在bug