import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# 1. 加载数据
df = pd.read_csv('chapter7/cleaned.tsv', delimiter='\t')

# 2. 数据预处理
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 将文本转换为BERT输入格式
encoded_data = tokenizer.batch_encode_plus(
    df['Cleaned Review'].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = torch.tensor(df['Liked'].values)

# 3. 划分训练集和测试集
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# 4. 加载预训练模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# 5. 训练设置
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

# 6. 训练函数
def train():
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss}")

# 7. 验证函数
def validate():
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds)
            true_labels.extend(labels)
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")

# 8. 训练模型
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train()
    validate()

# 9. 保存模型
model.save_pretrained('chapter7/sentiment_model')
tokenizer.save_pretrained('chapter7/sentiment_model')
