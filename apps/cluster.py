# -*- coding: utf-8 -*-
"""cluster.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
通过高斯混合模型（GMM）完成短文本聚类

:copyright: (c) 2021 by Zhichao Xia
:modified: 2021-07-21

"""
import jieba
from apps.common.mgp import MovieGroupProcess


# 加载停用词词典
with open("../data/chineseStopWords.txt", 'r', encoding='utf-8') as f:
    stopword = [i.strip() for i in f.readlines()]

# 加载语料
with open("C:/Users/xzc/Desktop/Origina_field.txt", encoding='utf-8') as file:
    data = [i.strip() for i in file.readlines()]

sentences = []  # 存放所有分词后的句子
vocab = set()  # 词表
# 分词去停用词
for sentence in data:
    cut_sent = [word for word in jieba.lcut(sentence) if word not in set(stopword)]
    sentences.append(cut_sent)
    for word in cut_sent:
        vocab.add(word)

mgp = MovieGroupProcess(K=50, alpha=0.1, beta=0.1, n_iters=30)
result = mgp.fit(sentences, len(vocab))  # 输入为分词后的句子列表+词表长度

# 展示聚类后的结果
# for idx, topic_words in enumerate(mgp.cluster_word_distribution):
#     sorted_words = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)
#     print("\n", idx, " ".join(["{}:{}".format(k, v) for k, v in sorted_words[:50]]))

# 按聚类类别排序结果
clusterlist = []
for idx, filed in enumerate(result):
    clusterlist.append((filed, idx))
clusterlists = sorted(clusterlist, key=lambda x: x[0], reverse=False)

# 将原始数据排序按照上一步排序后的结果依次写入文件
write_file = open("C:/Users/xzc/Desktop/result_field.txt", 'w', encoding='utf-8')
data_file = []
with open("C:/Users/xzc/Desktop/Origina_field.txt", encoding='utf-8') as file:
    for idx, each_data in enumerate(file.readlines()):
        data_file.append((idx, each_data))
for cluster in clusterlists:
    for cur in data_file:
        if cur[0] == cluster[1]:
            print(cur[1])
            write_file.write(cur[1])
print("写入完毕！！！")