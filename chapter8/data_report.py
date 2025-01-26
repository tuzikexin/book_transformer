import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chapter8.chapter8_train import CNNDailyMailDataset
import pandas as pd
from collections import Counter

def generate_raw_data_report(dataset):
    """生成原始数据报告"""
    report = "# Raw Data Report\n\n"
    
    # 数据集基本信息
    report += "## Dataset Info\n"
    report += f"- Name: cnn_dailymail\n"
    report += f"- Version: 3.0.0\n"
    report += f"- Splits: {list(dataset.keys())}\n"
    
    # 统计每个split的大小
    report += "\n## Dataset Size\n"
    report += "- Note: Using small subset for faster processing\n"
    for split in dataset.keys():
        report += f"- {split}: {len(dataset[split])} samples\n"
    
    # 分析文章长度分布
    report += "\n## Article Length Analysis\n"
    article_lengths = [len(article.split()) for article in dataset['train']['article']]
    report += f"- Average length: {sum(article_lengths)/len(article_lengths):.1f} words\n"
    report += f"- Min length: {min(article_lengths)} words\n"
    report += f"- Max length: {max(article_lengths)} words\n"
    
    # 分析摘要长度分布
    report += "\n## Summary Length Analysis\n"
    summary_lengths = [len(summary.split()) for summary in dataset['train']['highlights'][:1000]]
    report += f"- Average length: {sum(summary_lengths)/len(summary_lengths):.1f} words\n"
    report += f"- Min length: {min(summary_lengths)} words\n"
    report += f"- Max length: {max(summary_lengths)} words\n"
    
    return report

def generate_cleaned_data_report(dataset):
    """生成清洗后数据报告"""
    report = "# Cleaned Data Report\n\n"
    
    # 初始化数据集类
    ds = CNNDailyMailDataset()
    
    # 将列表转换为字典格式
    train_samples = list(dataset['train'])
    train_dict = {
        'article': [sample['article'] for sample in train_samples],
        'highlights': [sample['highlights'] for sample in train_samples]
    }
    train_data = ds.preprocess_data(train_dict)
    
    # 分析输入文章的长度
    report += "## Input Length Analysis\n"
    ids_lengths = [len(ids) for ids in train_data['input_ids']]
    report += f"- Average length: {sum(ids_lengths)/len(ids_lengths):.1f} tokens\n"
    report += f"- Min length: {min(ids_lengths)} tokens\n"
    report += f"- Max length: {max(ids_lengths)} tokens\n"
    
    # 分析labels长度
    report += "\n## Label Length Analysis\n"
    label_lengths = [len(ids) for ids in train_data['labels']]
    report += f"- Average length: {sum(label_lengths)/len(label_lengths):.1f} tokens\n"
    report += f"- Min length: {min(label_lengths)} tokens\n"
    report += f"- Max length: {max(label_lengths)} tokens\n"
    
    return report

def save_report(report, filename):
    """保存报告为markdown文件"""
    with open(filename, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    # 加载数据集
    ds = CNNDailyMailDataset()
    dataset = ds.load_data()
    
    train_dataset, val_dataset, test_dataset = ds.get_datasets()

    # 生成报告'    
    raw_report = generate_raw_data_report(dataset)
    cleaned_report = generate_cleaned_data_report(dataset)
    
    # 保存报告
    save_report(raw_report, 'chapter8/raw_data_report.md')
    save_report(cleaned_report, 'chapter8/cleaned_data_report.md')
    
    print("Reports generated successfully!")
