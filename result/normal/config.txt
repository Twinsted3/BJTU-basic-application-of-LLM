# 配置参数
class ModelConfig:
    data_path = "./src/input.txt"  # 数据文件路径
    block_size = 256               # 序列最大长度
    
    # 模型参数
    n_layer = 6        # Transformer层数
    n_head = 8         # 多头注意力的头数
    d_model = 384      # 嵌入维度
    n_ff = d_model * 4 # 前馈网络隐藏层维度
    dropout = 0.3

    # 训练参数
    is_train = True
    batch_size = 32
    lr = 1e-4
    epochs = 20
    weight_decay = 1e-2
    grad_clip = 1.0
    device = 'cuda:0'

    # 消融实验
    no_positional_encoding = False
    no_residual = False
    no_layernorm = False

