# -*- coding: utf-8 -*-
"""
feature_engineering_mul.py
~~~~~~~~~~~~~~~~~~~~~~
特征工程-多标签分类

:copyright: (c) 2019 by Ziqi Xiong
:modified: 2019-11-25
"""
import re
import json
import numpy as np
import joblib
from apps.common.utils import load_csv, load_json
from apps.common.lexicon import cut_words, cut_words_with_pos, cut_words_list, cut_words_with_pos_list

np.seterr(divide='ignore', invalid='ignore')


class FeatureEngineering(object):
    """特征工程
    """
    def __init__(self, model_path="", vocab_file="", stop_file=""):
        """"""
        if model_path:
            self.load_model(model_path)
        else:
            if vocab_file:
                self.vocabulary = {
                    line.split(":")[0].strip(): int(line.split(":")[1].strip()) for line in open(vocab_file).readlines()}
                self.vocabulary_inverse = {v:k for k, v in self.vocabulary.items()}
            else:
                # 构建特征词汇
                self.vocabulary = self.build_vocabulary()
                self.vocabulary_inverse = {}
            if stop_file:
                self.stop_words = {line.strip() for line in open(stop_file).readlines()}
            else:
                self.stop_words = {}
            self.term_docs = None
            self.cates = {}
            self.cates_inverse = {}
            self.size_docs = 0

    def save_model(self, model_path):
        """保存特征"""
        save_json = {"vocabulary": self.vocabulary,
                     "term_docs": self.term_docs.tolist(),
                     "cates": self.cates,
                     "size_docs": self.size_docs}
        with open(model_path, "w") as fobj:
            json.dump(save_json, fobj)

    def load_model(self, model_path):
        """载入模型"""
        with open(model_path, encoding="utf-8") as fobj:
            save_json = json.load(fobj)
            self.vocabulary = save_json.get("vocabulary", {})
            self.term_docs = np.array(save_json.get("term_docs", []))
            self.cates = save_json.get("cates", {})
            self.size_docs = save_json.get("size_docs", 0)
            assert self.vocabulary
            assert self.term_docs.any()
            assert self.cates
            assert self.size_docs != 0
            self.vocabulary_inverse = {v: k for k, v in self.vocabulary.items()}
            self.cates_inverse = {v: k for k, v in self.cates.items()}

    def remove_stop_words(self, item):
        """去掉停用词"""
        if not isinstance(item, str):
            return False
        # if len(item) < 1:
        #     return False
        if re.search("[\u4e00-\u9fa5]", item) is None:
            return False
        return True

    def my_analyzer_list(self, doc):
        """向量化分析器

        主要包括分词以及去除停用词
        """
        return list(filter(self.remove_stop_words, cut_words_list(doc)))

    def my_analyzer_pos_list(self, doc):
        """向量化分析器

        主要包括分词以及去除停用词
        """
        return list(filter(self.remove_stop_words, cut_words_with_pos_list(doc)))

    def build_vocabulary(self):
        """生成自定义的行业词表"""
        # 行业词汇表字典初始化
        industry_vacab_dict = {}
        # 词汇的总集合
        industry_vacab_set = set()
        fr = open("../common/industry_word_dict.txt", 'r+', encoding="utf-8")
        # 将读取的str转换为字典
        industry_word_dict = eval(fr.read())
        # 遍历行业名称字典
        for industry_cate in industry_word_dict:
            # 生成每个行业的词汇列表
            each_industry_cate_vacab_list = list(industry_word_dict[industry_cate])
            # 遍历不同行业中的所有名称
            for detail_indus in each_industry_cate_vacab_list:
                # 将各个词汇加入到词汇集合表中
                industry_vacab_set.add(detail_indus)
        # 生成行业词汇词典
        for num, word in enumerate(industry_vacab_set):
            industry_vacab_dict[word] = num
        return industry_vacab_dict

    def build_train(self, train_file, fmt='json', cate_field='cate',
                    cont_field='title,docTitle,docContent,content', split=','):
        """建立训练集"""
        if fmt == 'json':
            data, label = load_json(train_file, cate_field=cate_field, cont_field=cont_field)
        elif fmt == 'csv':
            data, label = load_csv(train_file, splitor=split)
        else:
            raise ValueError("Not supported args")
        data = [self.my_analyzer_list(item) for item in data]
        # print("打印data>>>>>>>>>>>>>>>>>>>>>>", data)
        self.term_docs = np.zeros(len(self.vocabulary))
        # 训练数据是数组
        data_train = np.zeros((len(data), len(self.vocabulary)))
        for idx, doc in enumerate(data):
            # doc_set = set()
            # term 是文本中的词汇
            for term in doc:
                # 在行业字典中键是词汇名称，值是对应的整型数字
                if term in self.vocabulary:
                    data_train[idx, self.vocabulary[term]] += 1
                    # if term not in doc_set:
                    #     doc_set.add(term)
                    #     self.term_docs[self.vocabulary[term]] += 1
        # 标签数据
        data_label = []
        for cate in label:
            # 如果键不已经存在于字典中，将会添加键并将值设为默认值
            self.cates.setdefault(cate, len(self.cates))
            self.cates_inverse[self.cates[cate]] = cate
            data_label.append(self.cates[cate])
        return data_train, np.array(data_label)

    def build_test(self, test_file, fmt='json', cate_field='cate',
                   cont_field='title,docTitle,docContent,content', split=','):
        """构建测试集"""
        if fmt == 'json':
            data, label = load_json(test_file, cate_field=cate_field, cont_field=cont_field)
        elif fmt == 'csv':
            data, label = load_csv(test_file, splitor=split)
        else:
            raise ValueError("Not supported args")
        # 构建测试数据
        data = [self.my_analyzer_list(item) for item in data]
        data_test = np.zeros((len(data), len(self.vocabulary)))
        for idx, doc in enumerate(data):
            # 遍历每篇文档中的单词
            for term in doc:
                # 如果该单词在行业词典中
                if term in self.vocabulary:
                    # 在行业字典中键是词汇名称，值是对应的整型数字
                    data_test[idx, self.vocabulary[term]] += 1
        if label is not None:
            data_label = []
            for cate in label:
                mul_cate_list = []
                for new_cate in cate.split(";"):
                    if new_cate in self.cates:
                        # 小列表中添加类别
                        mul_cate_list.append(self.cates[new_cate])
                    else:
                        raise ValueError("Unseen category in test!!![%s]", new_cate)
                data_label.append(mul_cate_list)
        else:
            data_label = None
        return data_test, np.array(data_label)

    def check_nan(self, data_train):
        """检查是否有空行

        Args:
            data_train: 词频矩阵ß

        Returns:
            (idxs): 包含空行位置锁定的array，无空行则为空
        """
        idxs = np.where(np.sum(data_train, axis=1) == 0)
        return idxs

    def handling_empty_line(self, data, idxs, label=None):
        """处理无特征文档

        Args:
            data: 数据
            label: 对应的标签
            idxs: 空行下标

        Returns:
            (data, label): 去掉空行的数据和标签
        """
        assert isinstance(data, np.ndarray)
        if label is not None:
            assert data.shape[0] == label.shape[0]
        mask = np.ones(data.shape[0], dtype=bool)
        mask[idxs] = False
        data = data[mask]
        if label is not None:
            label = label[mask]
        return data, label

    def transform_tfidf(self, data_train):
        """计算tfidf
        """
        idxs = np.where(np.sum(data_train, axis=1) == 0)
        if idxs[0].any():
            raise ValueError("Empty line found in data, PLEASE remove these line before training: ", idxs)
        idf = np.log((1 + self.size_docs) / (1 + self.term_docs)) + 1
        data_train = data_train * idf
        ret = data_train / np.sqrt(np.sum(data_train ** 2, axis=1)).reshape(data_train.shape[0], 1)
        return ret


if __name__ == '__main__':
    # fe = FeatureEngineering()
    # X, Y = fe.build_train("/Users/hongziki/tmp/feature_test", fmt="csv", split=",")
    # print(X)
    # print(fe.vocabulary)
    # print(fe.transform_tfidf(X))
    fe = FeatureEngineering()
    print(fe.vocabulary)
    # print(len(fe.vocabulary))
    # fe.build_train(["1,石油 是 一群 中国 人", "0,他们 是 一群 美国 人","0,123 234 43"], fmt='csv', split=",")
    print(fe.build_test(["1,我们 空调 加工 中国 人", "0,他们 是 一群 美国 人", "0,123 234 43"], fmt='csv', split=","))
    # fe.save_model("../model/fe.model")
    # fee = FeatureEngImp()
    # fee = fee.load_model("../model/fe.model")
    # print(fee.tfidf_vec.vocabulary_)

