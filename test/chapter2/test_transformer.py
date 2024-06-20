from chapter2.pytorch_transformer import (MultiHeadAttention, PositionwiseFeedForward, 
                                        AddNorm,LearnablePositionalEncoding,
                                        PositionalEncoding)
import pytest
import torch

# =================== Test cases for the MultiHeadAttention class ===================
@pytest.fixture
def setup_attention():
    batch_size = 2
    seq_len = 5
    word_emb_d = 128
    d_model = 512
    num_heads = 4
    
    multi_head_attn = MultiHeadAttention(word_emb_d=word_emb_d, d_model=d_model, num_heads=num_heads)
    dummy_input = torch.rand(batch_size, seq_len, word_emb_d)
    return multi_head_attn, dummy_input

def test_output_shape(setup_attention):
    multi_head_attn, dummy_input = setup_attention
    output, attn_weights = multi_head_attn(dummy_input, dummy_input, dummy_input)
    assert output.shape == (dummy_input.size(0), dummy_input.size(1), multi_head_attn.d_model)
    assert attn_weights.shape == (dummy_input.size(0), multi_head_attn.num_heads, dummy_input.size(1), dummy_input.size(1))


# =================== Test cases for the PositionwiseFeedForward class ===================
class TestPositionwiseFeedForward:
    @pytest.fixture
    def model(self):
        return PositionwiseFeedForward(d_model=512, d_ff=1024, dropout=0.1)

    @pytest.fixture
    def input_tensor(self):
        return torch.randn(10, 20, 512)  # batch size of 10, sequence length of 20, model dimension of 512

    def test_output_shape(self, model, input_tensor):
        output = model(input_tensor)
        print(type(input_tensor), "!!!!!!")
        assert output.shape == input_tensor.shape, "Output shape should match input shape"

    def test_forward_pass(self, model, input_tensor):
        output = model(input_tensor)
        assert isinstance(output, torch.Tensor), "Output must be a torch tensor"
        assert not torch.isnan(output).any(), "Output should not contain NaNs"

    def test_components(self, model):
        assert model.w_1.in_features == 512 and model.w_1.out_features == 1024, "First linear layer dimensions are incorrect"
        assert model.w_2.in_features == 1024 and model.w_2.out_features == 512, "Second linear layer dimensions are incorrect"



#=================== Test cases for the TestAddNorm class ===================
class TestAddNorm:
    @pytest.fixture
    def model(self):
        # Assuming a typical dimension size used in transformers
        return AddNorm(size=512)

    @pytest.fixture
    def input_tensor(self):
        # Create a tensor with batch size of 10, sequence length of 20, and feature dimension of 512
        return torch.randn(10, 20, 512), torch.randn(10, 20, 512)

    def test_output_shape(self, model, input_tensor):
        x, sublayer_output = input_tensor
        output = model(x, sublayer_output)
        assert output.shape == x.shape, "Output shape should match input shape"

    def test_normalization_effectiveness(self, model, input_tensor):
        x, sublayer_output = input_tensor
        output = model(x, sublayer_output)
        # Check that the mean is approximately 0 and std is approximately 1
        assert torch.allclose(output.mean(), torch.tensor(0.0), atol=1e-3), "Mean of output should be close to 0"
        assert torch.allclose(output.std(), torch.tensor(1.0), atol=1e-3), "Standard deviation of output should be close to 1"

    def test_functionality(self, model, input_tensor):
        x, sublayer_output = input_tensor
        output = model(x, sublayer_output)
        # Ensuring that output is indeed different from both inputs unless x and sublayer_output are zeros
        assert not torch.all(torch.eq(output, x)), "Output should not be identical to input x"
        assert not torch.all(torch.eq(output, sublayer_output)), "Output should not be identical to sublayer output"


#=================== Test cases for the PositionalEncoding class ===================
def test_positional_encoding_shape():
    d_model = 512
    max_len = 5000
    batch_size = 32
    seq_len = 100

    # 创建PositionalEncoding实例
    pe = PositionalEncoding(d_model=d_model, max_len=max_len)
    # 模拟输入数据
    x = torch.zeros(batch_size, seq_len, d_model)
    # 获取输出
    output = pe(x)

    # 检查输出形状
    assert output.shape == (batch_size, seq_len, d_model), "Output shape is incorrect"

def test_learnable_positional_encoding_shape():
    d_model = 512
    max_len = 5000
    batch_size = 32
    seq_len = 100

    # 创建LearnablePositionalEncoding实例
    lpe = LearnablePositionalEncoding(d_model=d_model, max_len=max_len)
    # 模拟输入数据
    x = torch.zeros(batch_size, seq_len, d_model)
    # 获取输出
    output = lpe(x)

    # 检查输出形状
    assert output.shape == (batch_size, seq_len, d_model), "Output shape is incorrect"
    # 检查位置编码是否为参数
    assert 'encoding' in [name for name, _ in lpe.named_parameters()], "Encoding should be a learnable parameter"

# 可以运行pytest来执行这些测试
