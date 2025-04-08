from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from itertools import chain
import jieba
import numpy as np

# 假设我们有以下文本数据
documents = [
    "邮件是电子邮件的意思",
    "电子邮件是邮件的一种",
    "学习TF-IDF需要理解邮件和电子邮件",
    "邮件和电子邮件在互联网上传输"
]

# 定义获取高频词的函数
def get_top_words(documents, top_num):
    all_words = []
    for doc in documents:
        words = jieba.cut(doc)
        all_words.extend(words)
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

# 定义特征提取函数，允许选择特征提取方式
def extract_features(documents, feature_method='tf_idf', top_n=100):
    if feature_method == 'tf':
        # 获取高频词
        top_words = get_top_words(documents, top_n)
        # 构建特征向量
        feature_matrix = []
        for doc in documents:
            words = jieba.cut(doc)
            word_count = Counter(words)
            feature_vector = [word_count[word] for word in top_words]
            feature_matrix.append(feature_vector)
        return np.array(feature_matrix)
    elif feature_method == 'tf_idf':
        # 使用 TfidfVectorizer 计算 TF-IDF 值
        vectorizer = TfidfVectorizer(analyzer=jieba.cut)
        feature_matrix = vectorizer.fit_transform(documents).toarray()
        return feature_matrix
    else:
        raise ValueError("Invalid feature method. Choose 'tf' or 'tf_idf'.")

# 选择特征提取方式
feature_method = 'tf_idf'  # 可以是 'tf' 或 'tf_idf'

# 提取特征
features = extract_features(documents, feature_method)

# 输出特征矩阵
print(features)