from transformers import pipeline
import torch

# 加载训练好的模型
classifier = pipeline(
    'sentiment-analysis',
    model='chapter7/sentiment_model',
    device=0 if torch.cuda.is_available() else -1
)

# 测试样本
test_samples = [
    "This product is absolutely amazing!",
    "I'm really disappointed with the quality.",
    "The service was okay, nothing special.",
]

# 进行预测
print("Testing sentiment analysis model...\n")
for sample in test_samples:
    result = classifier(sample)
    print(f"Text: {sample}")
    print(f"Prediction: {result[0]['label']} (confidence: {result[0]['score']:.4f})")
    print("-" * 50)
