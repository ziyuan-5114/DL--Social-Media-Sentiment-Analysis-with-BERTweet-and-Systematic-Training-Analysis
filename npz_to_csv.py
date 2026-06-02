# -*- coding: utf-8 -*-
"""
将 processed_data.npz 转换为可读的 CSV 文件
输出：train_text.csv, val_text.csv, test_text.csv
"""

import numpy as np
import pandas as pd

# 1. 加载 .npz 文件
data = np.load('processed_data.npz', allow_pickle=True)

# 查看包含的数组
print("文件中包含的数组:", data.files)

# 2. 提取数据
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

# vocab 和 label_map 是以对象数组保存的，需要用 .item() 取出字典
vocab = data['vocab'].item()        # 词到索引的映射
label_map = data['label_map'].item() # 标签到索引的映射

max_len = data['max_len']            # 最大序列长度

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")
print(f"词汇表大小: {len(vocab)}")
print(f"标签映射: {label_map}")

# 3. 构建索引到词的映射（用于还原文本）
idx_to_word = {idx: word for word, idx in vocab.items()}
# 注意：<PAD> 的索引是0，我们还原时跳过它

# 反向标签映射（从索引到标签名）
inv_label_map = {v: k for k, v in label_map.items()}

# 4. 定义一个函数，将索引序列还原为文本
def seq_to_text(seq):
    """
    将索引序列转换为单词字符串，忽略填充的0（<PAD>）
    """
    words = []
    for idx in seq:
        if idx == 0:  # <PAD>
            continue
        word = idx_to_word.get(idx, '<UNK>')
        words.append(word)
    return ' '.join(words)

# 5. 处理训练集
print("\n正在处理训练集...")
train_texts = [seq_to_text(seq) for seq in X_train]
train_label_names = [inv_label_map[y] for y in y_train]
train_df = pd.DataFrame({
    'text': train_texts,
    'label_index': y_train,
    'label_name': train_label_names
})
train_df.to_csv('train_text.csv', index=False, encoding='utf-8')
print(f"训练集已保存到 train_text.csv，共 {len(train_df)} 条")

# 6. 处理验证集
print("\n正在处理验证集...")
val_texts = [seq_to_text(seq) for seq in X_val]
val_label_names = [inv_label_map[y] for y in y_val]
val_df = pd.DataFrame({
    'text': val_texts,
    'label_index': y_val,
    'label_name': val_label_names
})
val_df.to_csv('val_text.csv', index=False, encoding='utf-8')
print(f"验证集已保存到 val_text.csv，共 {len(val_df)} 条")

# 7. 处理测试集
print("\n正在处理测试集...")
test_texts = [seq_to_text(seq) for seq in X_test]
test_label_names = [inv_label_map[y] for y in y_test]
test_df = pd.DataFrame({
    'text': test_texts,
    'label_index': y_test,
    'label_name': test_label_names
})
test_df.to_csv('test_text.csv', index=False, encoding='utf-8')
print(f"测试集已保存到 test_text.csv，共 {len(test_df)} 条")

# 8. （可选）合并所有数据集，添加一列标识
print("\n正在合并所有数据集...")
train_df['dataset'] = 'train'
val_df['dataset'] = 'val'
test_df['dataset'] = 'test'
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
all_df.to_csv('all_text.csv', index=False, encoding='utf-8')
print(f"合并后的数据集已保存到 all_text.csv，共 {len(all_df)} 条")

print("\n全部完成！")