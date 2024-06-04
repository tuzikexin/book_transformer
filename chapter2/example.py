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


# ============================== Decoder 使用示例 ==============================
# 假设我们有以下输入
batch_size = 2
seq_len_dec = 5
seq_len_enc = 6
d_model = 512
num_heads = 8
num_layers = 6

# 随机生成输入数据
dec_inputs = torch.randn(batch_size, seq_len_dec, d_model)
enc_outputs = torch.randn(batch_size, seq_len_enc, d_model)

# 随机生成掩码
dec_mask = torch.ones(batch_size, 1, seq_len_dec, seq_len_dec)  # Look-ahead mask
enc_mask = torch.ones(batch_size, 1, 1, seq_len_enc)  # Padding mask

# 初始化解码器
decoder = Decoder(d_model, num_heads, num_layers)

dec_output, dec_attn_weights, dec_enc_attn_weights = decoder(dec_inputs, enc_outputs, dec_mask, enc_mask)

print("Decoder Output Shape:", dec_output.shape)
print("Decoder Self-Attention Weights Shape:", [w.shape for w in dec_attn_weights])
print("Decoder-Encoder Attention Weights Shape:", [w.shape for w in dec_enc_attn_weights])