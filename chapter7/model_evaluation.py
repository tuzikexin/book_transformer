import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, learning_curve
from transformers import BertForSequenceClassification, BertTokenizer

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

def evaluate_robustness(model, dataloader, noise_levels=[0.0, 0.1, 0.2, 0.3]):
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
                
                # 添加噪声
                noisy_input = input_ids + torch.randn_like(input_ids) * noise
                
                outputs = model(noisy_input, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        results[noise] = accuracy
        print(f"Noise Level {noise:.1f} - Accuracy: {accuracy:.4f}")
    
    print("\nRobustness Evaluation Results:")
    for noise, acc in results.items():
        print(f"Noise Level {noise:.1f}: Accuracy = {acc:.4f}")
    
    # 绘制鲁棒性曲线
    plt.figure(figsize=(8, 5))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title('Model Robustness Evaluation')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    return results

def plot_learning_curve(model, dataloader, title="Learning Curve"):
    """绘制学习曲线"""
    input_data = []
    labels = []
    
    for batch in dataloader:
        input_data.append(batch[0].cpu().numpy())
        labels.append(batch[2].cpu().numpy())
    
    input_data = np.concatenate(input_data)
    labels = np.concatenate(labels)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, input_data, labels, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.show()

def run_evaluation(model_path, dataloader):
    """运行完整的评估流程"""
    print("Loading model...")
    model, _ = load_model(model_path)
    
    print("\n=== Stability Evaluation ===")
    stability_results = evaluate_stability(model, dataloader)
    
    print("\n=== Robustness Evaluation ===")
    robustness_results = evaluate_robustness(model, dataloader)
    
    print("\n=== Learning Curve Analysis ===")
    plot_learning_curve(model, dataloader)
    
    return {
        'stability': stability_results,
        'robustness': robustness_results
    }

if __name__ == "__main__":
    # 示例用法
    model_path = "chapter7/sentiment_model"
