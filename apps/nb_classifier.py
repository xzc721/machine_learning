# -*- coding: utf-8 -*-
"""nb_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本模块实现朴素贝叶斯分类器。

:copyright: (c) 2019 by Zhichao Xia
:modified: 2021-06-22

"""
import math
import numpy
from pprint import pprint
from collections import Counter
from apps.common.feature_engineering import FeatureEngineering


class NBCategorization:
    """朴素贝叶斯分类器"""
    def __init__(self):
        self.exclude_str = "[^\u4e00-\u9fa5]"
        self.model_x = None
        self.model_y = None

    def train(self, X, Y):
        """训练模型

        Args:
            X: (numpy.array), shape(文档数量, 词表长度)，训练数据
            Y: (numpy.array), shape(1, 文档数量)，类别标签，类别必须是[0,n]连续
        """
        total_docs = Y.shape[0]  # 数据总量
        cates_distrib = Counter(Y)  # 类别：数据量
        # shape(类别数量，词表长度)
        self.model_x = numpy.zeros((len(cates_distrib), X.shape[1]))
        self.model_y = numpy.zeros(len(cates_distrib))
        # 计算类别概率
        for cate, count in cates_distrib.items():
            self.model_y[cate] = math.log((count + 1)/(total_docs + len(cates_distrib)))
        # 计算词项概率
        # 数据载入
        for idx, data in enumerate(X):
            self.model_x[Y[idx]] += data  # shape(类别数，词表长度)
        # 计算对数概率 log((1+freqs)/(total_freqs+words_length))
        self.model_x = numpy.log((1+self.model_x)/(numpy.sum(self.model_x, axis=1).reshape(len(cates_distrib), 1) + X.shape[1]))

    def predicate_pure_prob(self, X):
        """预测 @TODO
        """
        assert self.model_x is not None and self.model_y is not None
        if not X.any():  # .any()表示：或操作，任意一个元素为True，输出为True。
            return []
        ret = numpy.matmul(X, self.model_x.T) + self.model_y

    def predicate_pure(self, X):
        """预测
        """
        assert self.model_x is not None and self.model_y is not None
        if not X.any():
            return []
        return numpy.argmax(numpy.matmul(X, self.model_x.T) + self.model_y, axis=1)

    def check_close(self, preds):
        """返回区分度不高的预测结果
        """
        if preds[0][1] - preds[1][1] < 2 or preds[1][1] - preds[2][1] < 2:
            return True
        return False

    def test(self, X, Y):
        """测试"""
        # 混淆矩阵
        confusion_m = numpy.zeros((self.model_y.shape[0], self.model_y.shape[0]))
        preds = self.predicate_pure(X)
        for idx, pred in enumerate(preds):
            confusion_m[Y[idx]][pred] += 1
        pprint(confusion_m)
        for idx in range(self.model_y.shape[0]):
            print("{}|| Precision: [{:.2%}], Recall: [{:.2%}]".format(
                idx,
                confusion_m[idx][idx]/sum(confusion_m[:, idx]),
                confusion_m[idx][idx]/sum(confusion_m[idx, :])))

    def save_model(self, model_file):
        """保存模型"""
        if self.model_x is not None and self.model_y is not None:
            with open(model_file, "w") as writef:
                writef.write("categories:\n")
                writef.write("{}\n".format(" ".join([str(i) for i in self.model_y])))
                writef.write("features:\n")
                for idx in range(self.model_x.shape[0]):
                    writef.write("{}\n".format(" ".join([str(i) for i in self.model_x[idx]])))

    def load_model(self, model_file):
        """"""
        with open(model_file, "r") as fobj:
            cont = fobj.read()
            categories, features = cont.split("features:")
            _, categories = categories.split("categories:")
            self.model_y = numpy.array([float(i) for i in categories.strip().split()])
            model_x = []
            for line in features.strip().split("\n"):
                model_x.append([float(i) for i in line.strip().split()])
            self.model_x = numpy.array(model_x)


if __name__ == "__main__":
    nb = NBCategorization()
    fe = FeatureEngineering()
    # 加载训练数据
    X_train, Y_train = fe.build_train("F:/项目/machine_learning/data/train_data.csv", fmt='csv', split='__label__')
    # 加载测试数据
    X_test, Y_test = fe.build_test("F:/项目/machine_learning/data/test_data.csv", fmt='csv', split='__label__')
    # X_valid, Y_valid = fe.build_test("F:/项目/machine_learning/data/test_data.csv", fmt='csv', split='__label__')
    # nb.train(X_train, Y_train)  # 训练打开
    # nb.save_model("../model/nb.model")  # 训练打开
    nb.load_model("../model/nb.model")  # 测试打开
    nb.test(X_test, Y_test)  # 测试打开
