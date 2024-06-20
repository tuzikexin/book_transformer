import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class ScaledDotProductAttention(nn.Module):
    # 缩放点积注意力层,用于计算Query,Key,Value之间的注意力权重
    # 输入为Q,K,V向量,维度为(batch_size,num_heads,seq_len,head_dim)
    # 首先计算Q与K的点积,对应论文公式中的QK^T
    # 使用根号下QK^T的维度大小对点积结果进行缩放(缩小张量的方差)
    # 如果提供了掩码mask,将掩码为0的位置的分数设为一个非常小的值,以无视这些位置
    # 对缩放的点积求softmax,得到最终的注意力权重
    # 将注意力权重与V向量相乘,得到加权后的注意力值,作为该层的输出
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        # q,k,v [batch_size, num_heads, seq_len, head_dim]
        head_dim = q.size(-1)

        # 得到scores/attn_weights大小: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        
        if mask is not None:
            # 对计算出的注意力分数加上掩码
            scores = scores.masked_fill(mask == 0, -1e9)
        # 对注意力分数计算softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 将注意力分数与Value相乘,得到指定输出维度
        # output:[batch_size, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    # 多头注意力层,与缩放点积注意力层的区别在于,对每个head进行了并行计算
    # 输入向量大小[batch_size, seq_len, word_emb_d]
    # 初始化时指定词镶嵌维度大小word_emb_d，模型维度d_model和head数量num_heads,根据d_model计算每个head的维度head_dim
    # 通过线性映射分别得到Q,K,V向量,它们的形状为[batch_size,num_heads,seq_len,head_dim]
    # 如果提供了掩码mask,就对其扩展成多头情况的掩码[batch_size,num_heads,seq_len,seq_len]
    # 对于每个head,分别计算Q,K,V的缩放点积注意力,得到每个head的输出和注意力权重
    # 将所有head的输出沿着head维度concatenate起来
    # 最终通过线性映射输出形状为[batch_size,seq_len,d_model]结果
    def __init__(self, d_model=512, num_heads=6, word_emb_d=None):
        super(MultiHeadAttention, self).__init__()
        if word_emb_d is None:
            word_emb_d = d_model
            
        # d_model 维度是所有head维度的总和，所以这里要确保d_model可以被head整除
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads

        # 线性映射分别得到Q,K,V向量
        self.q_linear = nn.Linear(word_emb_d, d_model) 
        self.v_linear = nn.Linear(word_emb_d, d_model)  
        self.k_linear = nn.Linear(word_emb_d, d_model)

        # 缩放点积运算层
        self.scale_dotp_atten = ScaledDotProductAttention()
        # 最终的线性输出层
        self.out_linear = nn.Linear(d_model, d_model)

    def head_split(self, all_heads_tensor):
        """
        将d_model按照头数分割
        : 输入all_heads_tensor: [batch_size, seq_leng, d_model]
        : 输出head_tensor: [batch_size, c, seq_len, head_dim]
        """
        batch_size, seq_len, _ = all_heads_tensor.size()

        # 变换维度为[batch_size, num_heads, seq_len, head_dim]
        head_tensor = all_heads_tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return head_tensor

    def forward(self, q, k, v, mask=None):
        # q, k, v 输入维度是[batch_size, seq_len, word_emb_d]
        batch_size, seq_len, _  = q.size()
        #      得到的新向量为[batch_size, seq_len, d_model]
        q = self.q_linear(q) 
        k = self.q_linear(k)
        v = self.q_linear(v)

        # 得到每一个head 的独立矩阵[batch_size, num_heads, seq_len, head_dim]
        q = self.head_split(q)
        k = self.head_split(k)
        v = self.head_split(v)
        
        # 将掩码mask扩展成多头情况,代表对于每个head的mask
        if mask is not None:
            mask = mask.unsqueeze(1)
            # mask : [batch_size, 1, seq_len, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        
        # 对每一个head进行缩放点积
        # attn_output:[batch_size, num_heads, seq_len, head_dim]
        # attn_weights:[batch_size, num_heads, seq_len, seq_len]
        attn_output, attn_weights = self.scale_dotp_atten(q, k, v, mask=mask)  # attn_weights 可以用来绘图做验证
        
        # 将head经过缩放点积注意力计算的结果从新拼接起来
        # 维度从[batch_size, head, seq_len, head_dim] 到 [batch_size, seq_len, head, head_dim] ->
        # 得到 out: [batch_size, seq_len, d_model]
        out = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 最后进过一个线性层输出结果[batch_size, seq_len, d_model]
        out = self.out_linear(out)
        return out, attn_weights


class PositionwiseFeedForward(nn.Module):
    # 前馈网络层,包含两个线性映射和一个ReLU激活
    # 初始化时指定输入维度d_model和内部维度d_ff以及dropout比例
    # 第一个线性层将d_model映射到d_ff,第二个线性层将d_ff映射回d_model
    # 前向传播时,先通过第一个线性层,然后对输出施加ReLU激活
    # 再通过第二个线性层,并对输出施加dropout
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 首先通过第一个线性层映射,并应用ReLU激活
        output = self.w_1(x)
        output = F.relu(output)
        
        # 然后通过第二个线性层映射,并执行dropout
        output = self.w_2(output)
        output = self.dropout(output)
        return output


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        # 设定一个小的常数，避免除零操作
        self.eps = eps

    def forward(self, x):
        # 计算输入x的最后一维的均值
        # '-1' 是指最后一个维度. 
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # 根据均值和方差进行归一化处理
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class AddNorm(nn.Module):
    def __init__(self, size, eps=1e-12):
        super(AddNorm, self).__init__()
        self.norm = LayerNorm(size, eps)

    def forward(self, x, sublayer_output):
        # 将输入x与子层输出通过相加的方式相连
        added = x + sublayer_output
        # 归一化处理
        out = self.norm(added)
        return out

class EncoderLayer(nn.Module):
# 编码器层,集成了多头注意力层和前馈网络层
# 最终返回归一化后的输出,以及注意力权重
    def __init__(self, d_model, num_heads):
    # 初始化时指定模型维度d_model和头数量num_heads
        super(EncoderLayer, self).__init__()

        #前向传播前,首先通过多头注意力层,得到注意力输出
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)

        # 并通过一个前馈网络层, 内部维度d_ff大小可以根据情况修改，这里定义为输入d_model 4倍大
        self.ff = PositionwiseFeedForward(d_model, d_model*4)
        
        # 使用两个LayerNorm层,分别对注意力输出和前馈网络输出进行归一化
        self.layernorms = nn.ModuleList([LayerNorm(d_model) for _ in range(2)])

    def forward(self, x, mask):
        # 首先通过多头注意力层，并应用掩码
        attn_output, attn_weights = self.multi_head_attn(x, x, x, mask)
        # 对多头注意力输出首先残差连接（注意力输出与输入X相加）然后施加LayerNorm
        out1 = self.layernorms[0](x + attn_output)
        # 然后通过前馈网络层
        ff_output = self.ff(out1)
        # 对前馈网络输出施加LayerNorm,然后残差连接
        out2 = self.layernorms[1](out1 + ff_output)
        
        return out2, attn_weights
    
class Encoder(nn.Module):
    # 初始化时指定模型参数d_model, num_heads和层数num_layers
    def __init__(self, d_model, num_heads, num_layers):
        super(Encoder, self).__init__()
        # 完整的Encoder模块,由多个EncoderLayer层组成（num_layers）
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

        self.layernorms = nn.ModuleList([LayerNorm(d_model) for _ in range(3)])

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
        self.norm = LayerNorm(d_model)

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


class PositionalEncoding(torch.nn.Module):
    # 位置编码层用于将序列的位置信息编码到embedding中
    # max_len是假设的一个句子最多包含5000个token
    # 使用sin/cos函数计算位置编码的值,公式为PE(pos,2i)=sin(pos/10000^(2i/d_model))
    # 返回大小[batch_size, max_len,d_model]
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 开始位置编码部分,先生成一个max_len * d_model 的矩阵[5000 * 512]
        # 5000是一个句子中最多的token数，512是一个token用多长的向量来表示，5000*512这个矩阵用于表示一个句子的信息
        encoding = torch.zeros(max_len, d_model)
        # 创建一个位置序列 position[max_len,1],即[5000,1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 先把括号内的分式求出来,pos是[5000,1],分母是[d_model/2],通过广播机制相乘后是[5000,256]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                              -(math.log(10000.0) / d_model))
        
        # 填充位置编码矩阵
        encoding[:, 0::2] = torch.sin(position * div_term)  # 使用正弦波填充偶数索引
        encoding[:, 1::2] = torch.cos(position * div_term)  # 使用余弦波填充奇数索引

        # 因为一个句子要做一次位置编码，一个batch中会有多个句子
        # 所以增加一维用来和输入的一个batch的数据相加时做广播
        # [max_len,d_model] -> [1, max_len,d_model]
        encoding = encoding.unsqueeze(0)

        # 注册位置编码为模块的buffer，这样它不会被视为模型参数,不会被更新！
        self.register_buffer('encoding', encoding)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]'''
        # 5000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        
        # 将位置编码添加到输入向量
        x = x + self.encoding[:, :x.size(1), :]
        # 进过drop 后得到[batch_size, seq_len, d_model], 保持和输入的形状相同
        out = self.dropout(x) 
        return out

class LearnablePositionalEncoding(torch.nn.Module):
    # 可学习位置的编码方式
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__() 
         # 创建可学习的位置编码参数
        self.encoding = torch.nn.Parameter(torch.zeros(1, max_len, d_model)) 
        
    def forward(self, x):
        # 将位置编码添加到输入的特征上
        out = x + self.encoding[:, :x.size(1), :]
        return out 

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



