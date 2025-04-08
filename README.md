# 项目名称

## 代码核心功能说明

1. **文本预处理**：
   - 使用 `jieba` 进行中文分词。
   - 过滤无效字符和停用词。
   - 提取文本中的有效词汇。

2. **特征提取**：
   - **高频词模式**：统计文本中高频词的出现次数作为特征。
   - **TF-IDF 模式**：使用 TF-IDF 算法提取文本特征，衡量词语的重要性。

3. **模型训练与预测**：
   - 使用多项式朴素贝叶斯分类器对文本数据进行分类。
   - 提供预测功能，对未知文本进行分类。

4. **灵活切换特征模式**：
   - 用户可以根据需求选择高频词模式或 TF-IDF 模式。
   - 通过简单的代码修改即可切换特征提取方式。


## 特征模式介绍

### 1. 高频词模式
**说明**：  
高频词模式通过统计文本中出现频率最高的单词作为特征。这种方法简单直观，适用于词汇分布较为集中的文本数据。

**公式**：  
$TF(t, d) = \frac{n_t}{N}$


**代码示例**：
```python
from collections import Counter
from itertools import chain

def get_top_words(top_num):
    all_words = []
    filename_list = [f'邮件_files/{i}.txt' for i in range(151)]  # 假设有 151 个文件
    for filename in filename_list:
        words = get_words(filename)
        all_words.append(words)
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

top_words = get_top_words(100)  # 获取前 100 个高频词
```

### 2. TF-IDF 模式
**说明**：  
TF-IDF（Term Frequency-Inverse Document Frequency）是一种统计方法，用于评估一个词语对于一个文档集合中的某一份文档的重要性。它通过计算词语在文档中的频率（TF）和逆文档频率（IDF）来衡量词语的重要性。

**公式**：  
$$\[ TF-IDF = TF \times IDF \]$$

**代码示例**：
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 假设 all_texts 是包含所有文本内容的列表
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(all_texts)
```

## 切换特征模式的方法

### 1. 使用高频词模式
在代码中，通过调用 get_top_words 函数获取高频词列表，并基于这些高频词构建特征向量。以下是完整的代码示例：

```python
# 获取高频词
top_words = get_top_words(100)  # 获取前 100 个高频词

# 构建特征向量
vector = []
for filename in filename_list:
    words = get_words(filename)
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
vector = np.array(vector)
```

### 2. 使用 TF-IDF 模式
在代码中，使用 `TfidfVectorizer` 类对文本数据进行向量化处理。以下是完整的代码示例：

```python
# 假设 all_texts 是包含所有文本内容的列表
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(all_texts)
```

## 使用方法
1. **安装依赖**：确保安装了 numpy、scikit-learn、jieba等库。
   ```bash
   pip install numpy scikit-learn jieba
   ```
2. **准备数据**：将文本文件放在指定目录中。
3. **选择特征模式**：根据需求选择高频词模式或 TF-IDF 模式，运行代码进行模型训练和预测。

## 示例代码
以下是完整的示例代码，展示如何在两种特征模式之间切换：

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from collections import Counter
from itertools import chain
import os
import re

# 定义获取文本单词的函数
def get_words(filename):
    words = []
    try:
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
                words.extend(jieba.cut(line))
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
    except Exception as e:
        print(f"读取文件 {filename} 时发生错误: {e}")
    return words

# 获取高频词
def get_top_words(top_num):
    all_words = []
    filename_list = [f'邮件_files/{i}.txt' for i in range(151)]  # 假设有 151 个文件
    for filename in filename_list:
        words = get_words(filename)
        all_words.append(words)
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

# 高频词模式
top_words = get_top_words(100)
vector = []
for filename in filename_list:
    words = get_words(filename)
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
vector = np.array(vector)

# TF-IDF 模式
all_texts = [" ".join(get_words(filename)) for filename in filename_list]
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(all_texts)

# 准备标签数据
labels = np.array([1] * 127 + [0] * 24)

# 训练模型
model = MultinomialNB()
model.fit(vector, labels)

# 定义预测函数
def predict(filename):
    words = get_words(filename)
    current_vector = np.array(
        tuple(map(lambda word: words.count(word), top_words)))
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'

# 测试预测函数
print('151.txt 分类情况:{}'.format(predict('邮件_files/151.txt')))
print('152.txt 分类情况:{}'.format(predict('邮件_files/152.txt')))
```
