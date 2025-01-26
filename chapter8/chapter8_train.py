from transformers import (
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
import evaluate
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import re
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


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
        
        # 定义一个小的样本作为演示
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
        labels = self.tokenizer(
            text_target=highlights,
            max_length=1024,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
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
        
        # 应用预处理函数
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

class TextSummarizationTrainer:
    def __init__(self, model_name='meta-llama/Llama-3.2-1B-Instruct'):
        torch.mps.empty_cache()  # 释放MPS缓存
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.metrics = {
            'rouge': evaluate.load('rouge', trust_remote_code=True),
            'bleu': evaluate.load('bleu')
        }

    def setup_model(self):
        """加载LLaMA模型和分词器"""
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            # max_position_embeddings=1024,
            # pad_token_id=0
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            # model_max_length=1024,
            # padding_side='right'
        )
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        return self.model, self.tokenizer

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        # 将 predictions 从 logits 转换为 token IDs
        if predictions.ndim == 3:  # 如果是 logits
            predictions = np.argmax(predictions, axis=-1)

        # -100 是用于忽略的 token ID。在解码 labels 之前，换为 tokenizer.pad_token_id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 计算ROUGE指标
        rouge_result = self.metrics['rouge'].compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # 计算BLEU指标
        bleu_result = self.metrics['bleu'].compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        return {
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeL'],
            'bleu': bleu_result['bleu']
        }

    def train(self, train_dataset, val_dataset):
        """训练模型"""
        training_args = TrainingArguments(
            output_dir='./chapter8/results',
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            eval_strategy="epoch",
            logging_dir='./logs',
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            use_cpu=True,
            fp16=True if torch.cuda.is_available() else False,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            logging_steps=100,
            report_to="none"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=lambda data: {
                'input_ids': torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(d['input_ids'], dtype=torch.long) for d in data],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                ),
                'attention_mask': torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(d['attention_mask'], dtype=torch.long) for d in data],
                    batch_first=True,
                    padding_value=0
                ),
                'labels': torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(d['labels'], dtype=torch.long) for d in data],
                    batch_first=True,
                    padding_value=-100  # HF中-100 指出token应忽略
                )
            },
            compute_metrics=self.compute_metrics
        )

        # 开始训练
        self.trainer.train()
        return self.trainer

    def evaluate(self, test_dataset):
        """评估模型"""
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        eval_results = self.trainer.evaluate(test_dataset)
        return eval_results

    def save_model(self, output_dir):
        """保存模型"""
        if self.trainer is None:
            raise ValueError("Model must be trained before saving")
        
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return output_dir

def run():
    # 初始化数据集
    print("Loading and preprocessing dataset...")
    dataset = CNNDailyMailDataset()
    train_dataset, val_dataset, test_dataset = dataset.get_datasets()
    
    # 初始化训练器
    print("Setting up model...")
    trainer = TextSummarizationTrainer()
    trainer.setup_model()
    
    # 开始训练
    print("Starting training...")
    trainer.train(train_dataset, val_dataset)
    
    # 评估模型
    print("Evaluating model...")
    eval_results = trainer.evaluate(test_dataset)
    print(f"Evaluation results: {eval_results}")
    
    # 保存模型
    print("Saving model...")
    model_path = trainer.save_model('./chapter8/results/saved_model')
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    run()
