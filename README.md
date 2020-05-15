# Speech Transformer: End-to-End ASR with Transformer
- 本项目基于transformer6*6层的基本结构构造的端到端的语音识别系统
- corpus表示数据集txt文件
- model_log表示模型存储文件和log文件
- src表示源代码文件
    - bin包括tran.py, recognition.py: 这两个文件通过修改内部参数可以直接运行,分别是训练以及测试功能
    - data 文件夹
        - dataloader.py 主要是负责数据集构成训练集合
        - dataset.py 主要是完成数据集的batch选取
        - load_corpus.py 主要是完成数据集文件的读取功能
    - solver文件夹
        - 主体训练流程逻辑代码
    - transformer文件夹
        - attention.py 自注意力机制计算过程
        - encoder.py 编码器计算过程
        - decoder.py 解码器计算过程
        - loss.py loss损失的计算
        - module.py 前馈网络FFNN的计算过程
        - optimizer.py 优化器迭代过程
        - transformer.py 整理网络架构设计
    - utils/const.py 一些常量的定义,如数据集目录,以及服务器的id选择
    - utils/params.py 数据集的选择的问题

# 模型训练: 
参数设置在train.py内部
```python
python3 train.py
```
    
# 模型测试:
参数设置在recognition.py内部
```python
python3 recognition.py
```

- 2017 google [Attention is All You Need]