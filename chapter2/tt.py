import matplotlib.pyplot as plt
import torch
from pytorch_transformer import (MultiHeadAttention, Decoder,Encoder,
                                 InputEmbeddings, PositionalEncoding, LearnablePositionalEncoding,
                                 PositionwiseFeedForward, AddNorm)


print("# ============================== Decoder 使用示例 ==============================")
# 假设我们有以下输入
batch_size = 2
seq_len_dec = 5
seq_len_enc = 6
d_model = 512
num_heads = 4
d_ff = 2048
num_layers = 6
dropout = 0.1

# 随机生成输入数据
dec_inputs = torch.randn(batch_size, seq_len_dec, d_model)
enc_outputs = torch.randn(batch_size, seq_len_enc, d_model)

# 随机生成掩码
src_mask = torch.ones(batch_size, seq_len_dec, seq_len_enc)
src_mask[:, 1, 1] = 0  # 添加一个掩码例子
tgt_mask = torch.ones(batch_size, 1, seq_len_dec)
tgt_mask[:, 0, 1] = 0 

# 初始化解码器
decoder = Decoder(d_model, num_heads, num_layers,d_ff,dropout)
dec_output, dec_attn_weights, dec_enc_attn_weights = decoder(dec_inputs, enc_outputs, src_mask, tgt_mask)

print("Decoder Output Shape:", dec_output.shape)
print(f"Decoder Self-Attention Weights has {len(dec_attn_weights)} layers, each one has shape {dec_attn_weights[0].shape}")
print(f"Decoder-Encoder Attention Weights has {len(dec_enc_attn_weights)} layers, each one has shape {dec_enc_attn_weights[0].shape}")
