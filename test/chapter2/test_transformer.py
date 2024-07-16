# 可以运行pytest来执行这些测试

from chapter2.pytorch_transformer import (
    ScaledDotProductAttention, MultiHeadAttention, 
    PositionwiseFeedForward, AddNorm, EncoderLayer,Encoder,
    LearnablePositionalEncoding, InputEmbeddings, PositionalEncoding,
    Decoder, DecoderLayer, Transformer,
    )
import pytest
import torch
import math


# 公共的输入数据fixture
@pytest.fixture
def common_input():
    batch_size, seq_len, d_model = 2, 10, 512
    dummy_input = torch.rand(batch_size, seq_len, d_model)
    return dummy_input, batch_size, seq_len, d_model

# =================== Test cases for the ScaledDotProductAttention class ===================
class TestScaledDotProductAttention():
    @pytest.fixture
    def input_tensors(self, common_input):
        _, batch_size, seq_len, d_model = common_input
        num_heads = 8
        head_dim = d_model // num_heads

        q = torch.rand(batch_size, num_heads, seq_len, head_dim)
        k = torch.rand(batch_size, num_heads, seq_len, head_dim)
        v = torch.rand(batch_size, num_heads, seq_len, head_dim)

        # 创建mask
        mask = torch.ones(batch_size, num_heads, seq_len, seq_len)
        mask[:, :, 1, 1] = 0  # 添加一个掩码例子

        return q, k, v, mask
    
    def test_scaled_dot_product_attention_no_dropout(self, input_tensors):
        q, k, v, mask = input_tensors
        attn_layer = ScaledDotProductAttention(dropout=None)
        
        output, attn_weights = attn_layer(q, k, v)
        
        assert output.shape == q.shape
        assert attn_weights.shape == (q.shape[0], q.shape[1], q.shape[2], q.shape[2])

    def test_scaled_dot_product_attention_with_dropout(self, input_tensors):
        q, k, v, mask = input_tensors
        attn_layer = ScaledDotProductAttention(dropout=0.1)
        
        output, attn_weights = attn_layer(q, k, v)
        
        assert output.shape == q.shape
        assert attn_weights.shape == (q.shape[0], q.shape[1], q.shape[2], q.shape[2])

    def test_scaled_dot_product_attention_with_mask(self, input_tensors):
        q, k, v, mask = input_tensors
        attn_layer = ScaledDotProductAttention(dropout=None)
        
        output, attn_weights = attn_layer(q, k, v, mask=mask)
        
        assert output.shape == q.shape
        assert attn_weights.shape == (q.shape[0], q.shape[1], q.shape[2], q.shape[2])
        assert (attn_weights[:, :, 1, 1] < 1e-6).all()  # 检查掩码是否生效

# =================== Test cases for the MultiHeadAttention class ===================
class TestMultiHeadAttention:
    @pytest.fixture
    def setup_attention(self, common_input):
        num_heads = 8
        dummy_input, batch_size, seq_len, d_model = common_input
        multi_head_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
        return multi_head_attn, dummy_input

    def test_output_shape(self, setup_attention):
        multi_head_attn, dummy_input = setup_attention
        output, attn_weights = multi_head_attn(dummy_input, dummy_input, dummy_input)
        assert output.shape == (dummy_input.size(0), dummy_input.size(1), multi_head_attn.d_model)
        assert attn_weights.shape == (dummy_input.size(0), multi_head_attn.num_heads, dummy_input.size(1), dummy_input.size(1))

# =================== Test cases for the PositionwiseFeedForward class ===================
class TestPositionwiseFeedForward:
    @pytest.fixture
    def fix_input(self, common_input):
        dummy_input, _, _, d_model = common_input
        d_ff = 1024
        return dummy_input, d_ff,d_model, PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)

    # @pytest.fixture
    # def input_tensor(self,common_input):
    #     c, batch_size, seq_len, d_model = common_input
    #     return torch.randn(10, 20, 512)  # batch size of 10, sequence length of 20, model dimension of 512

    def test_PositionwiseFeedForward_output_shape(self, fix_input):
        input_tensor, d_ff, d_model, model = fix_input
        output = model(input_tensor)
        assert output.shape == input_tensor.shape, "Output shape should match input shape"
        assert isinstance(output, torch.Tensor), "Output must be a torch tensor"
        assert not torch.isnan(output).any(), "Output should not contain NaNs"
        assert model.w_1.in_features == d_model and model.w_1.out_features == d_ff, "First linear layer dimensions are incorrect"
        assert model.w_2.in_features == d_ff and model.w_2.out_features == d_model, "Second linear layer dimensions are incorrect"

#=================== Test cases for the AddNorm class ===================
class TestAddNorm:
    @pytest.fixture
    def model(self):
        # Assuming a typical dimension size used in transformers
        return AddNorm(d_model=512)

    @pytest.fixture
    def input_tensor(self):
        # Create a tensor with batch size of 10, sequence length of 20, and feature dimension of 512
        return torch.randn(10, 20, 512), torch.randn(10, 20, 512)

    def test_AddNorm_output_shape(self, model, input_tensor):
        x, sublayer_output = input_tensor
        output = model(x, sublayer_output)
        assert output.shape == x.shape, "Output shape should match input shape"

    def test_normalization_effectiveness(self, model, input_tensor):
        x, sublayer_output = input_tensor
        output = model(x, sublayer_output)
        # Check that the mean is approximately 0 and std is approximately 1
        assert torch.allclose(output.mean(), torch.tensor(0.0), atol=1e-3), "Mean of output should be close to 0"
        assert torch.allclose(output.std(), torch.tensor(1.0), atol=1e-3), "Standard deviation of output should be close to 1"

    def test_AddNorm_functionality(self, model, input_tensor):
        x, sublayer_output = input_tensor
        output = model(x, sublayer_output)
        # Ensuring that output is indeed different from both inputs unless x and sublayer_output are zeros
        assert not torch.all(torch.eq(output, x)), "Output should not be identical to input x"
        assert not torch.all(torch.eq(output, sublayer_output)), "Output should not be identical to sublayer output"

#=================== Test cases for the PositionalEncoding class ===================
class TestPositionalEncoding: 
    max_len = 5000

    @pytest.fixture
    def ModelPositionalEncoding(self, common_input):
        _, _, _, d_model = common_input
        return PositionalEncoding(d_model= d_model, max_len= self.max_len)

    @pytest.fixture
    def ModelLearnablePositionalEncoding(self, common_input):
        _, _, _, d_model = common_input
        return LearnablePositionalEncoding(d_model= d_model, max_len= self.max_len)

    def test_positional_encoding_shape(self, ModelPositionalEncoding, common_input):
        input_tensor, batch_size, seq_len, d_model = common_input
        output = ModelPositionalEncoding(input_tensor)
        # 检查输出形状
        assert output.shape == (batch_size, seq_len, d_model), "Output shape is incorrect"

    def test_learnable_positional_encoding_shape(self, ModelLearnablePositionalEncoding, common_input):
        input_tensor, batch_size, seq_len, d_model = common_input
        output = ModelLearnablePositionalEncoding(input_tensor)

        # 检查输出形状
        assert output.shape == (batch_size, seq_len, d_model), "Output shape is incorrect"
        # 检查位置编码是否为参数
        assert 'encoding' in [name for name, _ in ModelLearnablePositionalEncoding.named_parameters()], "Encoding should be a learnable parameter"

#=================== Test cases for the InputEmbeddings class ===================
class TestInputEmbeddings: 
    @pytest.fixture
    def sample_input(self):
        return torch.tensor([[1, 2, 3], [4, 5, 6]])

    def test_forward_scaled_output(self, sample_input):
        d_model = 512
        vocab_size = 10000
        embedding_layer = InputEmbeddings(d_model, vocab_size)

        
        embeddings = embedding_layer.embedding(sample_input)
        scaled_embeddings = embeddings * math.sqrt(d_model)
        output = embedding_layer(sample_input)
        assert torch.allclose(output, scaled_embeddings)

        expected_shape = (sample_input.shape[0], sample_input.shape[1], d_model)
        assert output.shape == expected_shape

#=================== Test cases for the Encoder class ===================
class TestEncoder():
    @pytest.fixture
    def input_tensors(self, common_input):
        x, batch_size, seq_len, _ = common_input
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[0,0,1] = 0
        return x, mask

    def test_encoder_layer_forward(self, input_tensors):
        d_model = 512
        num_heads = 8
        d_ff = 2048
        dropout = 0.1
        x, mask = input_tensors

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        out, attn_weights = encoder_layer(x, mask)
        
        assert out.shape == x.shape
        assert attn_weights.shape == (x.shape[0], num_heads, x.shape[1], x.shape[1])

    def test_encoder_forward(self, input_tensors):
        d_model = 512
        num_heads = 8
        d_ff = 2048
        n_layers = 6
        dropout = 0.1
        x, mask = input_tensors

        encoder = Encoder(d_model, num_heads, n_layers, d_ff, dropout)
        out, attn_weights = encoder(x, mask)
        
        assert out.shape == x.shape
        assert len(attn_weights) == n_layers
        for attn_weight in attn_weights:
            assert attn_weight.shape == (x.shape[0],num_heads,x.shape[1], x.shape[1])

#=================== Test cases for the decoder class ===================
class TestDecoder:
    @pytest.fixture
    def sample_inputs(self):
        batch_size = 32
        seq_len = 10
        d_model = 512
        return torch.randn(batch_size, seq_len, d_model)

    @pytest.fixture
    def sample_enc_outputs(self):
        batch_size = 32
        seq_len = 10
        d_model = 512
        return torch.randn(batch_size, seq_len, d_model)

    @pytest.fixture
    def sample_masks(self):
        batch_size = 32
        seq_len = 10
        return torch.ones(batch_size, seq_len, seq_len)
    
    @pytest.fixture
    def decoder_sample(self):
        d_model = 512
        num_heads = 4
        n_layers = 3
        d_ff = 1024
        dropout = 0.1
        return (Decoder(d_model, num_heads, n_layers, d_ff, dropout),
                num_heads, n_layers)
    
    def test_decoder_initialization(self, decoder_sample):
        decoder, num_heads, n_layers = decoder_sample
        assert len(decoder.layers) == n_layers
        for layer in decoder.layers:
            assert isinstance(layer, DecoderLayer)

    def test_decoder_forward_output_shape(self, decoder_sample, sample_inputs, sample_enc_outputs, sample_masks):
        decoder, num_heads, n_layers = decoder_sample
        outputs, dec_attn_weights, dec_enc_attn_weights = decoder(sample_inputs, sample_enc_outputs, sample_masks, sample_masks)
        batch_size, seq_len, d_model = sample_inputs.size()
        assert outputs.size() == (batch_size, seq_len, d_model)
        assert len(dec_attn_weights) == n_layers
        assert len(dec_enc_attn_weights) == n_layers
        for attn_weights in dec_attn_weights:
            assert attn_weights.size() == (batch_size, num_heads, seq_len, seq_len)
        for enc_attn_weights in dec_enc_attn_weights:
            assert enc_attn_weights.size() == (batch_size, num_heads, seq_len, seq_len)

#=================== Test cases for the decoder class ===================
class TestTransformer:
    @pytest.fixture
    def transformer_model(self):
        src_vocab_size = 1345
        tgt_vocab_size = 234
        d_model = 512
        num_heads = 4
        num_encoder_layers = 3
        num_decoder_layers = 2
        d_ff = 256
        dropout = 0.1

        model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout)
        return model, num_heads, num_encoder_layers, num_decoder_layers

    @pytest.fixture
    def sample_inputs(self):
        batch_size = 7
        src_len = 54
        tgt_len = 11

        src = torch.randint(0, 1345, (batch_size, src_len))
        tgt = torch.randint(0, 234, (batch_size, tgt_len))
        return src, tgt

    def test_transformer_forward(self, transformer_model, sample_inputs):
        model, num_heads, num_encoder_layers, num_decoder_layers = transformer_model
        src, tgt = sample_inputs

        out, enc_attn_weights, dec_attn_weights, dec_enc_attn_weights = model(src, tgt)
        
        # 检查输出的形状是否正确
        assert out.shape == (src.size(0), tgt.size(1), model.fc_out.out_features)
        
        # 检查注意力权重的形状是否正确
        assert len(enc_attn_weights) == num_encoder_layers
        assert enc_attn_weights[0].shape == (src.size(0), num_heads, src.size(1), src.size(1))
        
        assert len(dec_attn_weights) == num_decoder_layers
        assert dec_attn_weights[0].shape == (src.size(0), num_heads, tgt.size(1), tgt.size(1))
        
        assert len(dec_enc_attn_weights) == num_decoder_layers
        assert dec_enc_attn_weights[0].shape == (src.size(0), num_heads, tgt.size(1), src.size(1))


if __name__ == "__main__":
    pytest.main([__file__])
