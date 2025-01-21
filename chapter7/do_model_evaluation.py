import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

def create_dataloader(data_path, batch_size=16):
    """创建评估用的DataLoader"""
    # 加载数据
    df = pd.read_csv(data_path, delimiter='\t')
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 编码数据
    encoded_data = tokenizer.batch_encode_plus(
        df['Cleaned Review'].tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # 创建TensorDataset
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(df['Liked'].values)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def load_model(model_path):
    """加载训练好的模型"""
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def evaluate_stability(model, dataloader, num_runs=5):
    """评估模型稳定性"""
    results = []
    print("Running stability evaluation...")
    
    for i in range(num_runs):
        model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0]
                attention_mask = batch[1]
                labels = batch[2]
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        results.append(accuracy)
        print(f"Run {i+1}/{num_runs} - Accuracy: {accuracy:.4f}")
    
    print(f"\nStability Evaluation Results:")
    print(f"Mean Accuracy: {np.mean(results):.4f}")
    print(f"Accuracy Std: {np.std(results):.4f}")
    return results

def evaluate_robustness(model, dataloader, outpath, noise_levels=[0.0, 0.1, 0.2, 0.3]):
    """评估模型鲁棒性"""
    results = {}
    print("Running robustness evaluation...")
    
    for noise in noise_levels:
        model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0]
                attention_mask = batch[1]
                labels = batch[2]
                
                # 添加噪声（转换为float32类型）
                noisy_input = input_ids.float() + torch.randn_like(input_ids.float()) * noise
                # 裁剪到有效范围并转换回long类型
                vocab_size = model.config.vocab_size
                noisy_input = torch.clamp(noisy_input, min=0, max=vocab_size-1).long()
                
                outputs = model(noisy_input, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        results[noise] = accuracy
        print(f"Noise Level {noise:.1f} - Accuracy: {accuracy:.4f}")
    acc_values = list(results.values()) 
    print("\nRobustness Evaluation Results:")
    print(f"Mean Accuracy: {np.mean(acc_values):.4f}")
    print(f"Accuracy Std: {np.std(acc_values):.4f}")

    # 绘制鲁棒性曲线
    plt.figure(figsize=(8, 5))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title('Model Robustness Evaluation')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(Path(outpath,'robustness_curve.png'))
    plt.close()
    
    return results

def plot_learning_curve(model, dataloader, outpath, title="Learning Curve"):
    """绘制学习曲线"""
    # 将数据分为训练集和验证集
    dataset = dataloader.dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    val_scores = []
    
    # 初始化优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = CrossEntropyLoss()
    
    for size in train_sizes:
        # 创建子训练集
        subset_size = int(size * train_size)
        subset_indices = torch.randperm(train_size)[:subset_size]
        subset = torch.utils.data.Subset(train_dataset, subset_indices)
        subset_loader = DataLoader(subset, batch_size=16, shuffle=True)
        
        # 训练模型
        model.train()
        for epoch in range(2):  # 每个子集训练2个epoch
            for batch in subset_loader:
                optimizer.zero_grad()
                input_ids = batch[0]
                attention_mask = batch[1]
                labels = batch[2]
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
        
        # 评估训练集和验证集
        model.eval()
        train_acc = evaluate_accuracy(model, subset_loader)
        val_acc = evaluate_accuracy(model, DataLoader(val_dataset, batch_size=16))
        
        train_scores.append(train_acc)
        val_scores.append(val_acc)
    
    # 绘制学习曲线
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    
    plt.grid()
    plt.plot(train_sizes * train_size, train_scores, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes * train_size, val_scores, 'o-', color="g",
             label="Validation score")
    
    plt.legend(loc="best")
    plt.savefig(Path(outpath,'learning_curve.png'))
    plt.close()

def evaluate_accuracy(model, dataloader):
    """评估模型准确率"""
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def save_results(results, outpath='chapter7/',filename='evaluation_results.txt'):
    """保存评估结果到文件"""
    with open(Path(outpath, filename), 'w') as f:
        f.write("=== Model Evaluation Results ===\n\n")
        f.write("Stability Evaluation:\n")
        f.write(f"Mean Accuracy: {np.mean(results['stability']):.4f}\n")
        f.write(f"Accuracy Std: {np.std(results['stability']):.4f}\n\n")
        
        f.write("Robustness Evaluation:\n")
        for noise, acc in results['robustness'].items():
            f.write(f"Noise Level {noise:.1f}: Accuracy = {acc:.4f}\n")

        acc_values = list(results['robustness'].values())
        f.write(f"\nMean Accuracy: {np.mean(acc_values):.4f}\n")
        f.write(f"Accuracy Std: {np.std(acc_values):.4f}\n")

def run_evaluation(model_path, dataloader, outpath='chapter7/'):
    """运行完整的评估流程"""
    print("Loading model...")
    model, _ = load_model(model_path)
    
    print("\n=== Stability Evaluation ===")
    stability_results = evaluate_stability(model, dataloader,num_runs=3)
    
    print("\n=== Robustness Evaluation ===")
    robustness_results = evaluate_robustness(model, dataloader, outpath)
    
    print("\n=== Learning Curve Analysis ===")
    plot_learning_curve(model=model, dataloader=dataloader, outpath=outpath)
    
    results = {
        'stability': stability_results,
        'robustness': robustness_results
    }
    
    # 保存结果
    save_results(results)
    print("\nResults saved to evaluation_results.txt")
    print("Learning curve saved to learning_curve.png")
    print("Robustness curve saved to robustness_curve.png")
    
    return results

if __name__ == "__main__":
    # 示例用法
    model_path = "chapter7/sentiment_model"
    data_path = "chapter7/cleaned.tsv"
    outpath = "chapter7/"
    
    # 创建dataloader
    dataloader = create_dataloader(data_path)
    
    # 运行评估
    results = run_evaluation(model_path, dataloader, outpath)
