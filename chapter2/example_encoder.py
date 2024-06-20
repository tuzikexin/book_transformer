import torch
from pytorch_transformer import Encoder

# ============================== Encoder 使用示例 ==============================
# 假设我们有以下输入
batch_size = 2
seq_len = 10
d_model = 512
num_heads = 8
num_layers = 6

# 随机生成输入数据
inputs = torch.randn(batch_size, seq_len, d_model)

# 随机生成掩码
mask = torch.ones(batch_size, 1, 1, seq_len)  # Padding mask

# 初始化编码器
encoder = Encoder(d_model, num_heads, num_layers)

# 前向传播
enc_output, attn_weights = encoder(inputs, mask)

print("Encoder Output Shape:", enc_output.shape)
print("Attention Weights Shape:", [w.shape for w in attn_weights])

