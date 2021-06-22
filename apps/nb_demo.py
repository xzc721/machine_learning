# -*- coding: utf-8 -*-
"""nb_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本模块负责申报类项目公文分类，类别包括：“资金扶持”、“科研项目”、“资质认定”、“科技奖励”、“其他”
贝叶斯定理：P(c|x) = (P(c)*P(x|c)) / P(x)
简单解析：因为对于分类问题来说，每个类别的分母相同，因此P(x)被统一归一化掉了
          也就是说，只需要计算类先验概率P(c)和样本x对于类c的类条件概率P(x|c)
朴素贝叶斯分类器 = 贝叶斯定理 + 属性条件独立性假设

:copyright: (c) 2021 by Zhichao Xia
:modified: 2021-06-18

:TODO:
"""
import math
import time
import codecs
import numpy as np
import jieba.analyse
from pprint import pprint
from sklearn.metrics import confusion_matrix


start = time.process_time()
# 加载停用词表
stopwords = set([line.strip() for line in codecs.open('./data/chineseStopWords.txt', 'r', 'utf-8').readlines()])

# 根据数据构建嵌套字典
total_docs = 0
categories = {}
with codecs.open("./data/sample_classification.csv", 'r', 'utf-8') as rf:
    for line in rf.readlines():
        if line.find("__label__") != -1:
            total_docs += 1
            title, label = line.strip().split("__label__")
            words = list(jieba.cut(title))
            words = [w for w in words if w not in stopwords]
            if label not in categories:
                categories[label] = {'doc_cnt': 1, 'words': {}}
            else:
                categories[label]['doc_cnt'] += 1
            for word in words:
                if word in categories[label]['words']:
                    categories[label]['words'][word] += 1
                else:
                    categories[label]['words'][word] = 1

# 计算字典中各类的值之和，即求类中词总数
for cla in categories:
    categories[cla]['word_sum'] = sum(categories[cla]['words'].values())
# vocabulary建立
vocabulary = set()
for label in categories.values():
    for word in label['words'].keys():
        vocabulary.add(word)

# 将单词扩充到每一个类别中，相当于完成平滑处理
for label in categories:
    for voc in vocabulary:
        if voc not in categories[label]['words'].keys():
            categories[label]['words'][voc] = 0
pprint(categories)
# 计算类先验概率P(c)
alpha = 1.0
classes = []  # 存放类别
for cla in categories:  # 得到类别数
    classes.append(cla)
for label in categories:
    categories[label]['prior_prob'] = (categories[label]['doc_cnt'] + alpha) / (total_docs + alpha * len(classes))
pprint(categories)
# 计算各word条件概率
for label in categories:
    for word, count in categories[label]['words'].items():
        prior_prob = ((count + alpha) / (categories[label]['word_sum'] + alpha * len(vocabulary)))  # 计算条件概率，P（x|c）
        categories[label]['words'][word] = {'count': count, 'cond_prob': prior_prob}
pprint(categories)

# 比较并输出结果分类
input_label = []
output_label = []
with codecs.open("./data/test_data.csv", 'r', 'utf-8') as rf:
    for line in rf.readlines():
        if line.find("__label__") != -1:
            test_title, true_label = line.strip().split("__label__")
            words = list(jieba.cut(test_title))
            test_words = [w for w in words if w not in stopwords]
            pred_value = -1000.0
            pred_label = ""
            for label in categories:
                current_class_prior = categories[label]['prior_prob']  # 类先验概率
                current_conditional_prob = 1.0  # 词条件概率
                for word in test_words:
                    if word in categories[label]['words'].keys():
                        current_conditional_prob += math.log(categories[label]['words'][word]['cond_prob'])
                tmp = current_conditional_prob + math.log(current_class_prior)
                #  比较并储存最大后验概率
                if tmp > pred_value:
                    pred_value = tmp
                    pred_label = label
            # 将真实值和预测值分别存入列表，以备混淆矩阵使用
            input_label.append(true_label)
            output_label.append(pred_label)
print(input_label)
print(output_label)
# 得到混淆矩阵
c = confusion_matrix(input_label, output_label, labels=["资质认定", "资金扶持", "科研项目", "科技奖励", "其他"])
print('混淆矩阵:\n', c)
L = np.tril(c, -1)  # 得到下三角矩阵
U = np.triu(c, 1)  # 得到上三角矩阵
sum_d = 0
recall = []
for i in range(len(classes)):  # 计算对角元素之和
    for j in range(i, len(classes)):
        sum_d += c[i][j]
FP = sum(sum(L))  # 下三角元素之和
FN = sum(sum(U))  # 上三角元素之和
print('准确率:', sum_d / sum(sum(c)))
print('召回率1:', c[0][0] / sum(c[0]))
print('召回率2:', c[1][1] / sum(c[1]))
print('召回率3:', c[2][2] / sum(c[2]))
print('召回率4:', c[3][3] / sum(c[3]))
print('召回率5:', c[4][4] / sum(c[4]))

end = time.process_time()
print('运行时间 %s秒' % (round(end - start, 2)))









