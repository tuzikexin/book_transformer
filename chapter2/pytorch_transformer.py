import torch
import torch.nn as nn
import math


class EncoderLayer(nn.Module):
# 编码器层,集成了多头注意力层和前馈网络层
# 最终返回归一化后的输出,以及注意力权重
    def __init__(self, d_model, num_heads):
    # 初始化时指定模型维度d_model和头数量num_heads
        super(EncoderLayer, self).__init__()

        #多头注意力层
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)

        # 前向传播前,首先通过多头注意力层,得到注意力输出
        # 将,并通过第一个LayerNorm层
        self.ff = PositionwiseFeedForward(d_model, d_model*4)
        
        # 使用两个LayerNorm层,分别对注意力输出和前馈网络输出进行归一化
        self.layernorms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

    def forward(self, x, mask):
        # 首先通过多头注意力层，并应用掩码
        attn_output, attn_weights = self.multi_head_attn(x, x, x, mask)
        # 对多头注意力输出施加LayerNorm,然后残差连接（注意力输出与输入X相加）
        out1 = self.layernorms[0](x + attn_output)
        
        # 然后通过前馈网络层
        ff_output = self.ff(out1)
        # 对前馈网络输出施加LayerNorm,然后残差连接
        out2 = self.layernorms[1](out1 + ff_output)
        
        return out2, attn_weights
    
class Encoder(nn.Module):
    # 完整的Encoder模块,由多个EncoderLayer层组成
    # 初始化时指定模型参数d_model,num_heads和层数num_layers
    def __init__(self, d_model, num_heads, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, x, mask):
        # 在forward函数中,输入依次通过每个EncoderLayer,并收集所有层的注意力权重
        # 最终返回最后一层的输出,以及所有层的注意力权重
        attn_weights = []
        for layer in self.layers:
            x, attn_weight = layer(x, mask)
            attn_weights.append(attn_weight)
        return x, attn_weights


class DecoderLayer(nn.Module):
    """
    # 解码器层,集成了两个多头注意力层和一个前馈网络层
    # 第一个多头注意力层用于编码当前序列,并施加掩码以忽略非法连接
    # 第二个多头注意力层用于将解码器输出与编码器输出进行合并
    # 两次多头注意力后,都要进行残差连接和LayerNorm归一化
    # 最后通过前馈网络层,并再次执行残差连接和LayerNorm
    # 返回归一化后的输出,以及解码器自注意力权重和编码器-解码器注意力权重
    """
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attn = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_model*4)

        self.layernorms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

    def forward(self, dec_inputs, enc_outputs, dec_mask, enc_mask):
        # 首先通过第一个多头注意力,对解码器输入序列进行掩蔽,保证不会关注后续位置
        dec_attn_output, dec_attn_weights = self.masked_multi_head_attn(dec_inputs, dec_inputs, dec_inputs, dec_mask)
        dec_output = self.layernorms[0](dec_attn_output + dec_inputs)

        # 然后通过第二个多头注意力层,结合编码器输出
        dec_enc_output, dec_enc_attn_weights = self.multi_head_attn(dec_output, enc_outputs, enc_outputs, enc_mask)
        dec_output = self.layernorms[1](dec_enc_output + dec_output)

        # 最后通过前馈网络层
        ff_output = self.ff(dec_output)
        dec_output = self.layernorms[2](ff_output + dec_output)

        return dec_output, dec_attn_weights, dec_enc_attn_weights
    

class Decoder(nn.Module):
    # 完整的Decoder模块,可以由多个DecoderLayer层组成

    def __init__(self, d_model, num_heads, num_layers):
        # 初始化时指定模型参数d_model,num_heads和层数num_layers  
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_inputs, enc_outputs, dec_mask, enc_mask):
        # 在forward函数中,输入依次通过每个DecoderLayer,并收集所有层的注意力权重
        outputs = dec_inputs
        dec_attn_weights = []
        dec_enc_attn_weights = []

        # 最后对所有层的输出进行LayerNorm归一化
        for layer in self.layers:
            outputs, attn_weights, enc_attn_weights = layer(outputs, enc_outputs, dec_mask, enc_mask)
            dec_attn_weights.append(attn_weights)
            dec_enc_attn_weights.append(enc_attn_weights)

        # 返回归一化后的输出,以及解码器自注意力权重和编码器-解码器注意力权重
        return self.norm(outputs), dec_attn_weights, dec_enc_attn_weights


class PositionalEncoding(nn.Module):
    # 位置编码层用于将序列的位置信息编码到embedding中
    # 初始化一个位置编码层,指定模型维度,dropout比率和最大序列长度
    # 创建一个存储位置编码的缓冲区pe,形状为(1,max_len,d_model)
    # 使用sin/cos函数计算位置编码的值,公式为PE(pos,2i)=sin(pos/10000^(2i/d_model))
    # 位置编码并入输入embedding,然后执行dropout操作
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    # 缩放点积注意力层,用于计算Query,Key,Value之间的注意力权重
    # 输入为Q,K,V向量,维度为(batch_size,head_num,seq_len,head_dim)
    # 首先计算Q与K的点积,对应论文公式中的QK^T
    # 使用根号下QK^T的维度大小对点积结果进行缩放(缩小张量的方差)
    # 如果提供了掩码mask,将掩码为0的位置的分数设为一个非常小的值,以无视这些位置
    # 对缩放的点积求softmax,得到最终的注意力权重
    # 将注意力权重与V向量相乘,得到加权后的注意力值,作为该层的输出
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        # 计算Query与所有Key的点积
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        # 对计算出的注意力分数加上掩码(之后解释)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 对注意力分数计算softmax,使用了注意力机制的核心公式
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 将注意力分数与Value相乘,并指定输出维度
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    # 多头注意力层,与上面的缩放点积注意力层的区别在于,对每个head进行了并行计算
    # 初始化时指定模型维度d_model和head数量num_heads,根据d_model计算每个head的维度head_dim
    # 通过线性映射分别得到Q,K,V向量,它们的形状为(batch_size,num_heads,seq_len,head_dim)
    # 如果提供了掩码mask,就对其扩展成多头情况的掩码(batch_size,num_heads,seq_len,seq_len)
    # 对于每个head,分别计算Q,K,V的缩放点积注意力,得到每个head的输出和注意力权重
    # 将所有head的输出沿着head维度concatenate起来,输出形状为(batch_size,seq_len,d_model)
    # 输出通过线性映射和dropout返回
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.o_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 对Q,K,V分别作线性映射,映射为(batch_size,num_heads,seq_len,head_dim)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 将掩码mask扩展成多头情况,代表对于每个head的mask
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 分别对每个head计算注意力,然后在concatenate所有head的输出
        attn_output, attn_weights = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # 将多头注意力输出作线性映射并执行dropout
        output = self.o_linear(attn_output)
        output = self.dropout(output)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    # 前馈网络层,包含两个线性映射和一个ReLU激活
    # 初始化时指定输入维度d_model和内部维度d_ff
    # 第一个线性层将d_model映射到d_ff,第二个线性层将d_ff映射回d_model
    # 前向传播时,先通过第一个线性层,然后对输出施加ReLU激活
    # 再通过第二个线性层,并对输出施加dropout
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 首先通过第一个线性层映射,并应用ReLU激活
        output = self.w_1(x)
        output = F.relu(output)
        
        # 然后通过第二个线性层映射,并执行dropout
        output = self.w_2(output)
        output = self.dropout(output)
        
        return output

class Transformer(nn.Module):
    # 完整的Transformer模型
    # 包含Encoder,Decoder,以及输入输出Embedding和线性映射层
    # 初始化时指定源语言和目标语言的词汇量,以及其他模型参数
    # 前向逻辑为:
    #   1. 通过Embedding层将输入源/目标序列转为词向量
    #   2. 通过PositionalEncoding层为Embedding加入位置信息
    #   3. 将embedded源序列输入Encoder,得到编码器输出    
    #   4. 将embedded目标序列与编码器输出一起输入Decoder
    #   5. 将解码器输出通过最后的线性层,得到词汇空间的输出    
    #   6. 返回模型输出以及编码器/解码器注意力权重
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_heads, num_layers)
        self.decoder = Decoder(d_model, num_heads, num_layers)
        self.out_linear = nn.Linear(d_model, tgt_vocab_size)
        
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.pos_encoder(self.src_emb(src))
        tgt_emb = self.pos_encoder(self.tgt_emb(tgt))
        enc_output, enc_attn_weights = self.encoder(src_emb, src_mask)
        dec_output, dec_attn_weights, dec_enc_attn_weights = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)
        output = self.out_linear(dec_output)
        return output, dec_attn_weights, dec_enc_attn_weights
    

class TextClassifier(nn.Module):
    # 使用Transformer作为编码器的文本分类模型
    # 初始化时指定词汇量大小,类别数量,以及Transformer参数
    # 在前向逻辑中:
    #   1. 将输入序列通过Transformer编码,获取序列的输出表示
    #   2. 取出编码器输出中的第一个token作为文档表示
    #   3. 将文档表示通过分类器的线性层输出分类分数
    # 可以根据输出分数使用交叉熵损失进行模型训练
    def __init__(self, vocab_size, num_classes, d_model=512, num_heads=8, num_layers=6):
        super(TextClassifier, self).__init__()
        self.transformer = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, inputs, masks=None):
        outputs, _, _ = self.transformer(inputs, inputs, masks, masks)
        outputs = outputs[:, 0, :]  # 只取序列的第一个token
        logits = self.classifier(outputs)
        return logits
