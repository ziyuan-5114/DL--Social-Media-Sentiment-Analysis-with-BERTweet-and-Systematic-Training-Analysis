# -*- coding: utf-8 -*-
"""
社交媒体情感分析数据预处理脚本
输入：包含 Year, Month, Day, Time of Tweet, text, sentiment, Platform 的CSV文件
输出：清洗后的训练/验证/测试集（序列化、标签编码、填充后）
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split

# ===== 确保 NLTK 分词资源已下载 =====
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("正在下载 punkt_tab 资源...")
    nltk.download('punkt_tab')
    # 如果下载 punkt_tab 后仍失败，再尝试下载 punkt（兼容旧版）
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
# ====================================

# 尝试导入 emoji 库（可选）
try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    print("提示：emoji库未安装，表情符号将被保留为原始字符（不影响模型训练）。")

# 下载NLTK分词数据（首次运行需取消注释）
# nltk.download('punkt')

# -------------------- 1. 加载数据（支持多种编码） --------------------
def load_data(file_path, encoding='gbk'):
    """
    读取CSV文件，返回DataFrame
    先尝试指定编码（默认gbk），失败则尝试latin-1
    """
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"成功使用 {encoding} 编码读取文件")
    except UnicodeDecodeError:
        print(f"使用编码 {encoding} 失败，尝试使用 latin-1 编码...")
        df = pd.read_csv(file_path, encoding='latin-1')
        print("成功使用 latin-1 编码读取文件（如有中文可能显示乱码，但文本内容通常可保留）")
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在，请检查路径")
        raise

    print("数据集形状:", df.shape)
    print("列名:", df.columns.tolist())
    print("前5行:\n", df.head())
    return df

# -------------------- 2. 探索性数据分析（EDA） --------------------
def eda(df):
    """简单探索：缺失值、情感分布、平台分布、文本长度"""
    print("\n--- 缺失值检查 ---")
    print(df.isnull().sum())

    print("\n--- 情感标签分布 ---")
    print(df['sentiment'].value_counts())

    print("\n--- 平台分布 ---")
    print(df['Platform'].value_counts())

    # 文本长度统计
    df['text_length'] = df['text'].astype(str).apply(len)
    print("\n--- 文本长度描述性统计 ---")
    print(df['text_length'].describe())

# -------------------- 3. 清洗情感标签 --------------------
def clean_sentiment(label):
    """
    将情感标签标准化：
      - 转为小写
      - 去除首尾空格和末尾标点符号（如 "positive!" -> "positive", "negative." -> "negative"）
    """
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    # 去除末尾的标点符号（! . , ; ? 等）
    label = re.sub(r'[!.,;:?]+$', '', label)
    return label

# -------------------- 4. 文本清洗函数 --------------------
def clean_text(text):
    """
    清洗单条文本
    步骤：
      - 如果安装了emoji库，将表情符号转换为文字描述（如 😊 -> smiling_face）
      - 移除URL
      - 将@用户名替换为特殊标记 <USER>
      - 移除#号但保留话题标签的文字
      - 合并多余空格，转为小写
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. 表情符号转换（可选）
    if HAS_EMOJI:
        text = emoji.demojize(text, delimiters=(" ", " "))

    # 2. 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. 替换@用户为<USER>
    text = re.sub(r'@\w+', '<USER>', text)

    # 4. 移除#号但保留话题标签的文字
    text = re.sub(r'#(\w+)', r'\1', text)

    # 5. 合并连续空格（不删除任何字符，保留表情符号等）
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. 转为小写
    text = text.lower()

    return text

# -------------------- 5. 分词与构建词汇表 --------------------
def tokenize_texts(texts, max_vocab_size=20000):
    """
    对文本列表进行分词，并构建词汇表（词->索引）
    返回：分词后的文本列表（每项为单词列表），以及词汇表、词频统计
    """
    tokenized = []
    word_counts = Counter()

    for text in texts:
        # 使用NLTK分词，它能够处理表情符号（将其作为单独token）
        tokens = word_tokenize(text)
        tokenized.append(tokens)
        word_counts.update(tokens)

    # 构建词汇表：保留最常见的max_vocab_size个词，其余标记为<UNK>
    vocab = {'<PAD>': 0, '<UNK>': 1}
    most_common = word_counts.most_common(max_vocab_size - 2)  # 减2保留特殊标记
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i

    return tokenized, vocab, word_counts

# -------------------- 6. 序列化与填充 --------------------
def encode_and_pad(tokenized_texts, vocab, max_len=100):
    """
    将分词后的文本转换为索引序列，并进行填充/截断
    """
    sequences = []
    for tokens in tokenized_texts:
        seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        if len(seq) > max_len:
            seq = seq[:max_len]  # 截断
        sequences.append(seq)

    # 填充到固定长度
    padded = np.array([seq + [0] * (max_len - len(seq)) for seq in sequences])
    return padded

# -------------------- 7. 自适应标签编码 --------------------
def encode_labels(labels):
    """
    根据情感列的实际值自动建立映射并编码
    返回：编码后的数组和映射字典
    """
    # 清洗标签（去除标点、小写）
    cleaned_labels = [clean_sentiment(l) for l in labels]
    # 获取唯一值，并排序以保证映射稳定
    unique_labels = sorted(set(cleaned_labels))
    print("发现的情感类别:", unique_labels)

    # 建立映射（例如 {'negative':0, 'neutral':1, 'positive':2}）
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print("标签映射:", label_map)

    # 编码
    encoded = [label_map[l] for l in cleaned_labels]
    return np.array(encoded), label_map

# -------------------- 8. 主流程 --------------------
def main():
    # ===== 请根据实际情况修改以下路径 =====
    file_path = "sentiment_analysis.csv"  # 替换为你的CSV文件名或绝对路径
    # =====================================

    # 1. 加载数据（默认尝试gbk编码）
    df = load_data(file_path, encoding='gbk')

    # 2. 简单EDA
    eda(df)

    # 3. 提取文本和标签
    texts = df['text'].astype(str).tolist()
    raw_labels = df['sentiment']  # 情感列

    # 4. 清洗文本
    print("\n--- 开始清洗文本 ---")
    cleaned_texts = [clean_text(t) for t in texts]
    print("清洗完成，示例：")
    for i in range(min(3, len(cleaned_texts))):
        print(f"原文本: {texts[i]}")
        print(f"清洗后: {cleaned_texts[i]}")
        print("---")

    # 5. 标签编码（自适应）
    print("\n--- 标签编码 ---")
    y, label_map = encode_labels(raw_labels)
    print(f"标签形状: {y.shape}")
    print("各类别样本数:", np.bincount(y))

    # 6. 分词与构建词汇表
    print("\n--- 分词与构建词汇表 ---")
    tokenized, vocab, word_counts = tokenize_texts(cleaned_texts, max_vocab_size=20000)
    print(f"词汇表大小: {len(vocab)}")
    print("最常见的10个词:", word_counts.most_common(10))

    # 7. 确定最大序列长度（使用95分位数，避免过长）
    lengths = [len(t) for t in tokenized]
    max_len = int(np.percentile(lengths, 95))
    print(f"设定序列最大长度: {max_len} (基于95分位数)")

    # 8. 序列化与填充
    print("\n--- 序列化与填充 ---")
    X = encode_and_pad(tokenized, vocab, max_len)
    print(f"特征矩阵形状: {X.shape}")

    # 9. 划分数据集（训练70%，验证15%，测试15%）
    print("\n--- 划分数据集 ---")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )  # 0.176 * 0.85 ≈ 0.15，使得验证集占总15%

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    # 10. 保存处理后的数据
    np.savez('processed_data.npz',
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             vocab=vocab, max_len=max_len, label_map=label_map)
    print("\n处理后的数据已保存为 processed_data.npz")

    # 可选：保存词汇表为文本文件，便于查看
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for word, idx in vocab.items():
            f.write(f"{word}\t{idx}\n")

    # 保存标签映射
    with open('label_map.txt', 'w', encoding='utf-8') as f:
        for label, idx in label_map.items():
            f.write(f"{label}\t{idx}\n")

if __name__ == "__main__":
    main()