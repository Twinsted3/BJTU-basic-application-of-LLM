# 大模型基础与应用期中作业

本项目实现了一个 **从零开始搭建的 Transformer 模型**，并使用 **Tiny Shakespeare 数据集** 进行训练与实验，包括 **消融实验**（如移除位置编码）和 **超参数调优实验**（如修改 `n_head`、`n_layer`、`d_model` 等参数），以研究模型性能变化。

---

## 📂 项目结构

<pre>
Project/
├── src/    # 数据集文件路径
│   ├── input.txt    # Tiny-shakespeare.txt
├── result/  # 结果文件，包含各次实验结果
│   ├── normal/      # 正常实验
│       ├── train_loss.csv   # train_loss文件
│       ├── res.txt          # 最后两轮的loss和采样结果
|       ├── config.txt       # 训练参数config文件
│   ├── no_pe/       # 消融实验，没有位置编码
│       ├── train_loss.csv
│       ├── res.txt
|       ├── config.txt
│     ……
├── run.py           # 实验运行主程序
├── config.py        # 实验运行设置
├── dataset.py       # 数据集类
├── model.py         # 模型类
├── ckpt.pt          # 最后一轮模型参数字典
├── best_ckpt.pt     # 最优模型参数字典  ### 两个ckpt内容传不上去github，故删去了
├── requirements.txt
└── README.md
</pre>

---

## 🧠 项目介绍

该项目旨在**从零实现 Transformer 架构**，包括：
- 词嵌入与位置编码；
- 多头自注意力机制；
- 前馈神经网络；
- 残差连接与层归一化；
- 训练与采样流程。

并在 Tiny Shakespeare 文本数据集上进行训练，通过不同实验对模型性能进行分析。

---

## 环境配置

请确保你的 Python 版本为 **3.8+**，然后在项目根目录下执行：

```bash
pip install -r requirements.txt
```

---

## 运行实验

在终端中进入项目目录：

```bash
cd 25125381-juwanglinbo/
python run.py
```

### 可修改参数
在 `config.py`中可以调整：

- n_head：多头注意力头数
- n_layer：Transformer层数
- d_model：隐藏维度
- dropout：dropout比例
- block_size：序列最大长度
- lr：学习率
- epoch：训练轮次
- batch_size：批次大小
- weight_decay：
- grad_clip：梯度裁剪
- device = 训练使用设备
- no_positional_embedding：不使用位置编码
- no_residual：不使用残差连接
- no_layernorm：不使用LayerNorm

---

## 实验结果

各实验结果存放在 result/ 目录下：

- normal/：标准Transformer实验
- no_pe/：移除位置编码
- no_layernorm/：移除LayerNorm
- no_residual/：移除残差连接
- moreHead/：更多注意力头，和更多维度的编码

其他文件夹表示调整不同参数的实验，每个实验包含：
- train_loss.csv：训练过程中每轮的 loss
- res.txt：最后两轮 loss 与采样文本结果
- config.txt：记录着训练config.py文件的内容

---

## 作者

25125381-琚王琳博

该项目仅用于大模型基础与应用课程考核内容。