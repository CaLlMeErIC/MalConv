# 目录
- [目录](#目录)
- [MalConv描述](#MalConv描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)
# MalConv描述

MalConv是一种卷积神经网络，其特点是读取原始文件的每一个字节作为输入特征，其特点是计算量和内存用量能够根据序列长度而高效地扩展，在检查整个文件的时候能同时考虑到本地和全局上下文
以及在分析标记为恶意软件的时候能够提供更好的解释能力。

[论文](https://arxiv.org/abs/1710.09435)：Edward Raff et al. 2018. Malware Detection by Eating a Whole EXE


# 数据集
使用的数据集黑样本来自kaggle Microsoft Malware Classification Challenge。
## [kaggle](<https://www.kaggle.com/competitions/malware-classification/data>)
- 数据集大小：37.89 GB 这个数据集共包含了九类恶意软件
- 数据格式：恶意软件原始文件
-注意事项：需要准备一个标签文件，其中以以字典的形式存储不同软件恶意软件md5对应的标签，用于黑白二分类时，默认是存在字典key中的md5为恶意软件md5
由于白样本涉及版权问题，需模型使用者自己搜集


# 环境要求

- 硬件（CPU）
    - 使用CPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 运行单机训练示例：
python train.py [train_samples]
# python train.py [train_samples]

# 运行评估示例
python eval.py [test_samples]
# python eval.py [test_samples]
```

# 脚本说明

## 脚本及样例代码

```md
├── datas
│   ├── label_dict.json //标签文件
│   ├── test_samples //测试样本文件夹
│   └── train_samples //训练样本文件夹
├── src
│   ├── args.py //参数配置文件
│   ├── dataset.py //数据集处理文件
│   ├── environment.py //环境配置文件
│   ├── feature.py //特征处理文件
│   ├── model.py //Malconv网络模型
│   ├── utils.py //文件操作函数
├── README.md //模型相关说明
├── predict.py //测试脚本
└── train.py //训练脚本
```



## 脚本参数

在args.py中可以同时配置训练参数、评估参数及模型导出参数。

  ```python
  # common_config
  'seed':1,#设置全局随机种子
  'mode':'graph',#设置运行模式
  'target': 'CPU', # 运行设备
  'device_id': 0, # 用于训练或评估数据集的设备ID
  'device_num':1,#设置设备数量
  'rank_id':1,#设置优先级

  # train_config
  'train_dir': './datas/train' # 训练文件夹的路径
  'test_dir': ''./datas/test'', # 测试文件夹的路径
  'label_dict': './datas/label_dict.json', # 标签字典文件
  'epoch': 1, # 训练轮数
  'lr': 0.0001, # 优化器学习率
  'ckpt_dir': './checkpoint', # checkpoint文件夹路径
  'batch_size': 8, # 一次读取数据大小

  # test_config
  'model': None, #用于预测的时候指定模型文件，如果未指定的话会在checkpoint文件夹下寻找ckpt文件读取 
  'output_file': 'result.txt', # 生成结果文件的文件名，其形式为md5对应为恶意软件的概率
  ```

更多配置细节请参考脚本`args.py`。

## 训练过程

### 单机训练

- CPU环境运行

```python
# 运行单机训练示例：
python train.py [train_samples]
# python train.py [train_samples]

# 运行评估示例
python eval.py [test_samples]
# python eval.py [test_samples]
```

  上述python命令将在后台运行，

  训练结束后，您可在存储的文件夹（./checkpoint）下找到生成的模型文件：

## 推理过程

### 推理

- 在CPU环境下评估

  在运行以下命令之前，请检查用于推理的检查点和json文件路径，并设置输出预测结果文件的路径。

```python
# CPU
python eval.py [test_samples] [output_file]
```

  上述python命令将在后台运行，结果将保存在outfile中，其形式是md5:为恶意软件概率的字典json文件


# 模型描述

## 性能

### 训练性能

在使用单cpu，epoch=5,batch_size=8的情况下能达到约95%的正确率

# 随机情况说明

在args.py中，设置了随机种子。

# reference

**[Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435)**

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。