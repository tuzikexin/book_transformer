import unittest
import torch
from chapter7.sentiment_analysis_model import (
    train,
    validate,
    model,
    tokenizer,
    train_dataloader,
    val_dataloader
)

class TestSentimentAnalysis(unittest.TestCase):
    
    def test_data_loading(self):
        # 测试数据加载是否成功
        self.assertIsNotNone(train_dataloader)
        self.assertIsNotNone(val_dataloader)
        self.assertGreater(len(train_dataloader.dataset), 0)
        self.assertGreater(len(val_dataloader.dataset), 0)

    def test_model_initialization(self):
        # 测试模型初始化
        self.assertIsNotNone(model)
        self.assertEqual(model.config.num_labels, 2)

    def test_tokenizer(self):
        # 测试tokenizer
        test_text = "This is a test sentence"
        encoded = tokenizer(test_text, return_tensors='pt')
        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)

    def test_training(self):
        # 测试训练过程
        initial_loss = validate()
        train()
        final_loss = validate()
        self.assertLess(final_loss, initial_loss)

    def test_validation(self):
        # 测试验证过程
        loss = validate()
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

if __name__ == '__main__':
    unittest.main()
