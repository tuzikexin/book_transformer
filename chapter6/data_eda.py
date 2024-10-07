import pandas as pd
import os
from pathlib import Path

# 读取数据
current_path = os.path.realpath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path)+os.path.sep+".")
data_path = Path(father_path, "fakenewskdd2020")
df = pd.read_csv(Path(data_path, 'train.csv'), sep='\t', encoding='utf-8')

# 数据概括
print(df.describe())
print("")

# 处理缺失值
print("#nr of missing values")
print(df.isnull().sum())
print("")

# 查看是否有重复的数据
print(f"#nr of duplicated records {df.duplicated().sum()}")
print("")

# 正标签和负标签数据比值
label = df["label"].value_counts()
print(f"The label 0: {label.iloc[0] / (label.iloc[0] + label.iloc[1]) * 100} % ")
print(f"The label 1: {label.iloc[1] / (label.iloc[0] + label.iloc[1]) * 100} % ")
print("")
