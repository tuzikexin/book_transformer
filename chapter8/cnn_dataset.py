from datasets import load_dataset
from transformers import AutoTokenizer
import re

class CNNDailyMailDataset:
    def __init__(self, model_name='meta-llama/Llama-3.2-1B-Instruct'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.raw_dataset = None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_data(self):
        """加载cnn_dailymail数据集
        返回：包含train/validation/test的小样本数据集
        """
        raw_dataset = load_dataset('cnn_dailymail', '3.0.0')
        self.raw_dataset = raw_dataset
        
        # 定义各数据集样本数量
        train_size = 10
        val_size = 3
        test_size = 2
        
        from datasets import DatasetDict
        small_dataset = DatasetDict({
            'train': raw_dataset['train'].select(range(train_size)),
            'validation': raw_dataset['validation'].select(range(val_size)),
            'test': raw_dataset['test'].select(range(test_size))
        })
        self.dataset = small_dataset
        
        print(f"Loaded dataset with {train_size} train, {val_size} validation, and {test_size} test samples")
        return self.dataset

    def clean_text(self, text):
        """文本清洗"""
        # 去除特殊字符和多余空格
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # 去除非ASCII字符
        text = text.strip()
        return text

    def preprocess_data(self, examples):
        """数据预处理"""
        # 清洗文章和摘要
        articles = [self.clean_text(article) for article in examples['article']]
        highlights = [self.clean_text(highlight) for highlight in examples['highlights']]
        
        # Tokenize articles with fixed length
        inputs = self.tokenizer(
            articles,
            max_length=1024,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize highlights
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                highlights,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
        # Create model inputs
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']
        }
        return model_inputs

    def prepare_datasets(self):
        """准备训练、验证和测试集"""
        if self.dataset is None:
            self.load_data()
        
        # 应用预处理
        tokenized_datasets = self.dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=['article', 'highlights', 'id']
        )
        
        # 数据集划分
        self.train_dataset = tokenized_datasets['train']
        self.val_dataset = tokenized_datasets['validation']
        self.test_dataset = tokenized_datasets['test']
        
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_datasets(self):
        """获取预处理后的数据集"""
        if self.train_dataset is None:
            self.prepare_datasets()
        return self.train_dataset, self.val_dataset, self.test_dataset
