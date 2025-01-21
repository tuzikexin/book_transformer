import pandas as pd

# 1. 数据概览
print("=== 数据概览 ===")
df = pd.read_csv('chapter7/restaurant_reviews.tsv', delimiter='\t')
print(f"总样本数: {len(df)}")
print(f"正面评论数: {len(df[df.Liked == 1])}")
print(f"负面评论数: {len(df[df.Liked == 0])}")

# 2. 数据质量检查
print("\n=== 数据质量检查 ===")
print("缺失值检查:")
print(df.isnull().sum())

print("\n重复值检查:")
print(f"重复样本数: {df.duplicated().sum()}")

print("\n评论长度统计:")
df['Review Length'] = df.Review.apply(len)
print(df['Review Length'].describe())

# 3. 数据清理
print("\n=== 数据清理 ===")
# 删除重复值
df = df.drop_duplicates()
print(f"删除重复值后样本数: {len(df)}")

# 处理缺失值（如果有）
df = df.dropna()
print(f"删除缺失值后样本数: {len(df)}")

# 文本清理
import re

def clean_text(text):
    # 转换为小写
    text = text.lower()
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Cleaned Review'] = df.Review.apply(clean_text)
print("\n文本清理示例:")
print(df[['Review', 'Cleaned Review']].head())

# 4. 保存清理后的数据
df.to_csv('chapter7/cleaned_restaurant_reviews.tsv', sep='\t', index=False)
print("\n清理后的数据已保存为 cleaned_restaurant_reviews.tsv")
