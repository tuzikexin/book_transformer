import torch
from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader


# 自定义TextClassificationDataset类,用于从文本和标签列表构建Dataset
# 重载__len__和__getitem__方法,以支持数据加载器的索引和取数操作

# 加载训练和测试数据,构建词汇表
# 将文本序列转换为数字索引形式
# 使用DataLoader从Dataset中加载批量数据

# 在GPU或CPU上创建模型实例
# 使用交叉熵损失函数和Adam优化器

# 训练循环:
#   1. 从数据加载器中获取一批数据
#   2. 将数据传入模型获取预测输出
#   3. 计算输出和标签之间的损失
#   4. 反向传播更新模型参数

# 评估循环:
#   1. 从测试数据加载器获取一批数据
#   2. 将数据传入模型获取预测输出
#   3. 统计预测正确的样本数量
#   4. 计算测试集准确率


# 自定义数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

# 加载和预处理数据
train_texts = [...] # 训练文本数据列表
train_labels = [...] # 训练标签列表
test_texts = [...] # 测试文本数据列表 
test_labels = [...] # 测试标签列表

# 构建词汇表
vocab = build_vocab(train_texts + test_texts)
vocab_size = len(vocab)

# 文本序列到数字索引的转换
train_data = encode_texts(train_texts, vocab)
test_data = encode_texts(test_texts, vocab)

# 创建数据集和数据加载器
train_dataset = TextClassificationDataset(train_data, train_labels)
test_dataset = TextClassificationDataset(test_data, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型初始化
num_classes = len(set(train_labels))
model = TextClassifier(vocab_size, num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 模型训练
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * texts.size(0)

    train_loss = train_loss / len(train_dataset)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')

# 模型评估
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for texts, labels in test_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')
# 自定义数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

# 加载和预处理数据
train_texts = [...] # 训练文本数据列表
train_labels = [...] # 训练标签列表
test_texts = [...] # 测试文本数据列表 
test_labels = [...] # 测试标签列表

# 构建词汇表
vocab = build_vocab(train_texts + test_texts)
vocab_size = len(vocab)

# 文本序列到数字索引的转换
train_data = encode_texts(train_texts, vocab)
test_data = encode_texts(test_texts, vocab)

# 创建数据集和数据加载器
train_dataset = TextClassificationDataset(train_data, train_labels)
test_dataset = TextClassificationDataset(test_data, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型初始化
num_classes = len(set(train_labels))
model = TextClassifier(vocab_size, num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 模型训练
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * texts.size(0)

    train_loss = train_loss / len(train_dataset)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')

# 模型评估
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for texts, labels in test_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')
