import matplotlib.pyplot as plt
import torch
from pytorch_transformer import (MultiHeadAttention, Decoder,Encoder,
                                 InputEmbeddings, PositionalEncoding, LearnablePositionalEncoding,
                                 PositionwiseFeedForward, AddNorm)


print("# ============================ MultiHeadAttention 使用示例: ============================")
batch_size = 2
seq_len = 5
word_emb_d = 256
d_model = 512
num_heads = 4
dummy_input = torch.rand(batch_size, seq_len, word_emb_d)

multi_head_attn = MultiHeadAttention(word_emb_d=word_emb_d, d_model=d_model, num_heads=num_heads)
output, attn_weights = multi_head_attn(dummy_input, dummy_input, dummy_input)

print("Output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)

print("# ============================== InputEmbeddings 使用示例 ==============================")
# 假设我们有以下输入
d_model = 512
vocab_size = 10000

# 创建 InputEmbeddings 类的实例
embedding_layer = InputEmbeddings(d_model, vocab_size)

# 创建形状为 (batch_size, seq_len) 的示例输入张量
sample_input = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 将示例输入传递通过嵌入层
output = embedding_layer(sample_input)

# 输出形状将为 (batch_size, seq_len, d_model)
print(output.shape)  # 输出: torch.Size([2, 3, 512])

print("# ============================ PositionalEncoding usage: ============================")
batch_size = 2
d_model = 512  # 定义嵌入的维度
seq_length = 100  # 定义序列长度

# 初始化模型
fixed_pos_encoder = PositionalEncoding(d_model)
learnable_pos_encoder = LearnablePositionalEncoding(d_model)

# 生成假数据
dummy_input = torch.randn(batch_size, seq_length, d_model)  # 生成随机数据作为输入

# 应用位置编码
fixed_encoded_output = fixed_pos_encoder(dummy_input)  # 应用固定位置编码
learnable_encoded_output = learnable_pos_encoder(dummy_input)  # 应用可学习位置编码

# 可视化位置编码
def plot_positional_encodings(pos_enc, title):
    plt.figure(figsize=(15, 10))
    # plt.imshow(pos_enc.detach().numpy(), cmap='viridis')
    plt.imshow(pos_enc.detach().numpy(),cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Dimension', rotation=270)
    plt.ylabel('Position', rotation=270)
    plt.savefig(f'{title}.jpg')
    plt.close()

plot_positional_encodings(fixed_pos_encoder.encoding.squeeze(), "Positional Encoding Visualization")
plot_positional_encodings(learnable_pos_encoder.encoding.squeeze().detach(), "Learnable Positional Encoding")

print("# ============================ Example usage PositionwiseFeedForward: ============================")
batch_size = 2
seq_len = 50
d_model = 512
d_ff = 1024
dummy_input = torch.rand(batch_size, seq_len, d_model)

positionwise_ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
output = positionwise_ff(dummy_input)
print("Output shape:", output.shape)

print("# ============================ Example usage AddNorm: ============================")
batch_size = 2
seq_len = 50
d_model = 512
d_ff = 1024
dummy_input = torch.rand(batch_size, seq_len, d_model)
add_norm = AddNorm(d_model=d_model)
output = positionwise_ff(dummy_input)
print("Output shape:", output.shape)

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
