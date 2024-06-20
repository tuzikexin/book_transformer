import matplotlib.pyplot as plt
from pytorch_transformer import PositionalEncoding, LearnablePositionalEncoding
import torch

# ============================ Example usage: ============================
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
