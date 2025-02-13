import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer
from transformer import (MultiHeadAttention, Decoder,Encoder,Transformer,
                                 InputEmbeddings, PositionalEncoding, LearnablePositionalEncoding,
                                 PositionwiseFeedForward, AddNorm)


print("# ============================ MultiHeadAttention 使用示例: ============================")
# 假设我们有以下输入
batch_size = 2
seq_len = 5
word_emb_d = 256
d_model = 512
num_heads = 4
dummy_input = torch.rand(batch_size, seq_len, word_emb_d)

multi_head_attn = MultiHeadAttention(word_emb_d=word_emb_d, d_model=d_model, num_heads=num_heads, dropout=0.1)
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

print("# ============================== encoder 使用示例 ==============================")
# 假设我们有以下输入
batch_size = 2
seq_len = 5
d_model = 512
num_heads = 2
d_ff = 2048
n_layers = 6
dropout = 0.1

# 随机生成输入数据
inputs = torch.randn(batch_size, seq_len, d_model)

# 随机生成掩码
mask = torch.zeros(batch_size, 1, seq_len)
mask[:, 0, 1] = 1  # 添加一个掩码例子

# 初始化编码器
encoder = Encoder(d_model, num_heads, n_layers, d_ff,dropout)

# 前向传播
enc_output, attn_weights = encoder(inputs, mask)

print("Encoder Output Shape:", enc_output.shape)
print(f"Encoder Self-Attention Weights has {len(attn_weights)} layers, each one has shape {attn_weights[0].shape}")

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
src_mask = torch.ones(batch_size, 1, seq_len_enc)
src_mask[:, 0, 1] = 0  # 添加一个掩码例子
tgt_mask = torch.ones(batch_size, seq_len_dec, seq_len_dec)
tgt_mask[:, 0, 1] = 0 

# 初始化解码器
decoder = Decoder(d_model, num_heads, num_layers,d_ff,dropout)
dec_output, dec_attn_weights, dec_enc_attn_weights = decoder(dec_inputs, enc_outputs, src_mask, tgt_mask)

print("Decoder Output Shape:", dec_output.shape)
print(f"Decoder Self-Attention Weights has {len(dec_attn_weights)} layers, each one has shape {dec_attn_weights[0].shape}")
print(f"Decoder-Encoder Attention Weights has {len(dec_enc_attn_weights)} layers, each one has shape {dec_enc_attn_weights[0].shape}")

print("# ============================== Transformer 使用示例 ==============================")
# 假设我们有以下输入
src_vocab_size = 1000
tgt_vocab_size = 200
d_model = 512
num_heads = 4
num_encoder_layers = 3
num_decoder_layers = 3
d_ff = 1024
dropout = 0.1
batch_size = 5
src_len = 20
tgt_len = 7

src = torch.randint(0, src_vocab_size, (batch_size, src_len))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout)
out, enc_attn_weights, dec_attn_weights, _ = model(src, tgt)

print(
    f"""
    Input nr_encoder_layer: {num_encoder_layers}, nr_decoder_layer: {num_decoder_layers},
    batch:{batch_size}, src_length:{src_len}, target_len:{tgt_len}, target_voc:{tgt_vocab_size}
    """)

print(f"The number of encoder layer: {len(enc_attn_weights)}")
print(f"The encoder self-attention scores shape: {enc_attn_weights[0].shape}") # [batch_size, head, src_len, src_len]
print(f"The decoder corss attention scores shape: {dec_attn_weights[0].shape}") # [batch_size, head, tgt_len, tgt_len]
print(f"The number of decoder layer: {len(dec_attn_weights)}")
print(f"Final output has shape: {out.shape}")  # [batch_size, tgt_len, tgt_vocab_size]

print("# ========================== example for tokenizer & mask ==========================")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 示例文本
text = "Transformers are great."

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 对文本进行分词
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# 将令牌转换为输入ID
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Input IDs:", input_ids)

print("# ========================== example for tokenizer & mask ==========================")
# 自定义mask函数
def create_masks(inputs, pad_token_id):
    mask = (inputs != pad_token_id).float()
    return mask

# 示例文本
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentences = ["Transformers are great.", 
             "They have revolutionized NLP."]

# 使用tokenizer对一个文本list添加特殊标记并创建注意力掩码
encoded_input = tokenizer(
    sentences,
    add_special_tokens=True,  # 添加 [CLS] 和 [SEP] 标记
    max_length=10,  # 填充和截断到最大长度
    padding='max_length',  # 填充到最大长度
    return_attention_mask=True,  # 返回注意力掩码
    return_tensors='pt',  # 返回PyTorch张量
    truncation=True,
)

print("Encoded Input:")
for k,v in encoded_input.items():
    print(f"   {k}: {v}")

print("# ========================== example for Dataloader & mask ==========================")
