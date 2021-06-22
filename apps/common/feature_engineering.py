# -*- coding: utf-8 -*-
"""featuren_engineering.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
特征工程

:copyright: (c) 2019 by Zhichao Xia
:modified: 2021-06-18
"""
import re
import json
import numpy as np
import joblib
from apps.common.utils import load_csv, load_json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from apps.common.lexicon import cut_words, cut_words_with_pos, cut_words_list, cut_words_with_pos_list


np.seterr(divide='ignore', invalid='ignore')  # 设置浮点错误的处理方式为ignore（忽略）


class FeatureEngImp(object):
    """特征工程

    Notes:
        基于sklearn的特征工程较为简洁，但是其模型会自动省略掉空行，导致向量化后的矩阵行数与原始数据对应
        不上，故暂未使用。
    """
    def __init__(self, vocab_file=None, stop_file=None, max_df=.95, min_df=1):
        """"""
        vocabulary = None
        stop_words = None
        if vocab_file is not None:
            vocabulary = {
                line.split(":")[0].strip(): int(line.split(":")[1].strip()) for line in open(vocab_file).readlines()}
        if stop_file is not None:
            stop_words = [line.strip() for line in open(stop_file).readlines()]
        self.count_vec = CountVectorizer(vocabulary=vocabulary, stop_words=stop_words, max_df=max_df, min_df=min_df,
                                         analyzer=self.my_analyzer_list)
        self.tfidf_vec = TfidfVectorizer(vocabulary=vocabulary, stop_words=stop_words, max_df=max_df, min_df=min_df,
                                         analyzer=self.my_analyzer_list)

    def save_model(self, model_path):
        """保存模型

        Args:
            model_path: 模型存储路径
        """
        joblib.dump(self, model_path)

    def load_model(self, model_path):
        """载入模型

        Args:
            model_path: 模型路径
        """
        return joblib.load(model_path)

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

    def handling_empty_line(self, data, label=None):
        """处理无特征文档

        Args:
            data: 数据
            label: 对应的标签

        Returns:
            (data, label): 去掉空行的数据和标签
        """
        assert isinstance(data, np.ndarray)
        if label is not None:
            assert data.shape[0] == label.shape[0]
        mask = np.ones(data.shape[0], dtype=bool)
        idxs = np.where(np.sum(data, axis=1) == 0)
        mask[idxs] = False
        data = data[mask]
        if label is not None:
            label = label[mask]
        return data, label

    def build_train(self, train_file, tfidf_flag=True, fmt='json', cate_field='cate',
                    cont_field='title,docTitle,docContent,content', split=','):
        """建立训练集
        """
        if fmt == 'json':
            data_train, label_train = load_json(train_file, cate_field=cate_field, cont_field=cont_field)
        elif fmt == 'csv':
            data_train, label_train = load_csv(train_file, splitor=split)
        else:
            raise ValueError("Not supported args")
        data_train_tfidf = self.tfidf_vec.fit_transform(data_train)
        data_train_count = self.count_vec.fit_transform(data_train)
        if data_train_count.shape[0] != len(data_train):
            raise ValueError("Empty lines found when transfering to count vector")
        if tfidf_flag:
            return data_train_tfidf, np.array(label_train)
        return data_train_count, np.array(label_train)

    def build_test(self, test_file, tfidf_flag=True, fmt='json', cate_field='cate',
                    cont_field='title,docTitle,docContent,content', split=','):
        """建立训练集
        """
        if fmt == 'json':
            data_test, label_test = load_json(test_file, cate_field=cate_field, cont_field=cont_field)
        elif fmt == 'csv':
            data_test, label_test = load_csv(test_file, splitor=split)
        else:
            raise ValueError("Not supported args")
        if tfidf_flag:
            data_tfidf = self.tfidf_vec.fit_transform(data_test)
            if data_tfidf.shape[0] != len(data_test):
                raise ValueError("Empty lines found when transfering to count vector")
            return data_tfidf, np.array(label_test)
        data_count = self.count_vec.fit_transform(data_test)
        if data_count.shape[0] != len(data_test):
            raise ValueError("Empty lines found when transfering to count vector")
        return data_count, np.array(label_test)

    def build_simple_data(self, data_file, tfidf_flag=True, fmt='json',
                          cont_field='title,docTitle,docContent,content', split=','):
        """构建一个普通的数据集（不包含类别信息）

        Args:
            data_file: 数据集文件路径或者数据列表
            tfidf_flag: 是否需要转化成tfidf向量
            fmt: 如果是数据集文件，该字段知名文件格式
            cont_field: 内容字段key

        Returns:
            (data): tfidf或tf形式的向量
        """
        if fmt == 'json':
            data, _ = load_json(data_file, cate_field=None, cont_field=cont_field)
        elif fmt == 'csv':
            data, _ = load_csv(data_file, splitor=None)
        else:
            raise ValueError("Not supported args")
        if tfidf_flag:
            data = self.tfidf_vec.transform(data)
        else:
            data = self.count_vec.transform(data)
        return data


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
                self.vocabulary_inverse = {v: k for k, v in self.vocabulary.items()}
            else:
                self.vocabulary = {}
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
            self.vocabulary = save_json.get("vocabulary", {})  # 词典
            self.term_docs = np.array(save_json.get("term_docs", []))
            self.cates = save_json.get("cates", {})  # 类别
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

    def build_vocabulary(self, data):
        """建立词表"""
        tmp_term_freqs = {}
        for line in data:
            for item in line:
                # 统计词频
                if item not in self.stop_words:
                    tmp_term_freqs.__setitem__(item, tmp_term_freqs.get(item, 0)+1)
            self.size_docs += 1
        for term, _ in [item for item in tmp_term_freqs.items() if item[1] >= 1]:
            self.vocabulary.setdefault(term, len(self.vocabulary))
            self.vocabulary_inverse.setdefault(self.vocabulary[term], term)

    def build_train(self, train_file, fmt='json', cate_field='cate',
                    cont_field='title,docTitle,docContent,content', split=','):
        """建立训练集"""
        if fmt == 'json':
            data, label = load_json(train_file, cate_field=cate_field, cont_field=cont_field)
        elif fmt == 'csv':
            data, label = load_csv(train_file, splitor=split)
        else:
            raise ValueError("Not supported args")
        data = [self.my_analyzer_list(item) for item in data]  # 分词，去停用词后的数据列表
        if not self.vocabulary:
            self.build_vocabulary(data)
        self.term_docs = np.zeros(len(self.vocabulary))
        data_train = np.zeros((len(data), len(self.vocabulary)))
        for idx, doc in enumerate(data):
            doc_set = set()
            for term in doc:
                if term in self.vocabulary:
                    data_train[idx, self.vocabulary[term]] += 1
                    if term not in doc_set:
                        doc_set.add(term)
                        self.term_docs[self.vocabulary[term]] += 1
        data_label = []
        for cate in label:
            self.cates.setdefault(cate, len(self.cates))
            self.cates_inverse[self.cates[cate]] = cate
            data_label.append(self.cates[cate])
        print(self.cates_inverse)
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
        data = [self.my_analyzer_list(item) for item in data]
        data_test = np.zeros((len(data), len(self.vocabulary)))
        for idx, doc in enumerate(data):
            for term in doc:
                if term in self.vocabulary:
                    data_test[idx, self.vocabulary[term]] += 1
        if label is not None:
            data_label = []
            for cate in label:
                if cate in self.cates:
                    data_label.append(self.cates[cate])
                else:
                    raise ValueError("Unseen category in test!!![%s]", cate)
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
    a = fe.build_train(["1,我们 是 一群 中国 人", "0,他们 是 一群 美国 人","0,123 234 43"], fmt='csv', split=",")
    print(a)
    # fe.save_model("../model/fe.model")
    # fee = FeatureEngImp()
    # fee = fee.load_model("../model/fe.model")
    # print(fee.tfidf_vec.vocabulary_)

