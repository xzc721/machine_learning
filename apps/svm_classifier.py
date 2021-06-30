# -*- coding: utf-8 -*-
"""svm_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本模块实现支持向量机分类器
设置validate_pure函数中detail=True，可打印分类错误的数据

:copyright: (c) 2019 by Zhichao Xia
:modified: 2021-06-22

"""
import re
import numpy
import pprint
import joblib
from sklearn.svm import SVC
from apps.common.feature_engineering import FeatureEngineering


class MyEngineering(FeatureEngineering):
    """自定义停用词特征工程
    """
    def remove_stop_words(self, item):
        """去掉停用词"""
        if not isinstance(item, str):
            return False
        if len(item) < 2:
            return False
        if re.search(r"[\u4e00-\u9fa5]", item) is None:
            return False
        return True


class SVMModel:
    """基于SVM的分类模型
    """
    def __init__(self, model_file="", feat_file="", vocab_file="", stop_file=""):
        """初始化

        Args:
            model_file (str) : svm模型文件路径
            feat_file (str) : 特征工程模型文件路径
        """
        if model_file and feat_file:
            self.load_model(model_file, feat_file)
        else:
            if vocab_file:
                self.fe = MyEngineering(vocab_file=vocab_file)
            elif stop_file:
                self.fe = MyEngineering(stop_file=stop_file)
            else:
                self.fe = MyEngineering()
            self.model = SVC(kernel='linear', probability=True, class_weight='balanced', verbose=True)

    def save_model(self, model_path, feat_path):
        """保存模型

        Args:
            model_path (str) : svm模型文件路径
            feat_path (str) : 特征工程模型文件路径
        """
        assert self.model is not None
        joblib.dump(self.model, model_path)
        self.fe.save_model(feat_path)

    def load_model(self, model_path, feat_path):
        """载入模型

        Args:
            model_path (str) : svm模型文件路径
            feat_path (str) : 特征工程模型文件路径
        """
        self.fe = FeatureEngineering(model_path=feat_path)
        self.model = joblib.load(model_path)
        assert self.model is not None

    def train(self, train_file, fmt='json', split='__label__'):
        """从原始数据构建模型

        Args:
            train_file (str) : 训练数据文件路径
            fmt (str) : train_file中内容的格式，目前支持两种“json”和“csv”, 当时“csv”时需同时制定分隔符“split”（见下）
            split (str) : 分隔符，格式为“csv”时需要据此分割数据和标签
        """
        train_data, train_label = self.fe.build_train(train_file, fmt=fmt, split=split)
        idx = self.fe.check_nan(train_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before training: ", idx)
            train_data, train_label = self.fe.handling_empty_line(train_data, label=train_label, idxs=idx)
        train_tfidf = self.fe.transform_tfidf(train_data)
        self.train_pure(train_tfidf, train_label)

    def predicate(self, test_data):
        """预测原始数据（暂未使用）

        Args:
            test_data (str or list) : 测试数据文件或数据列表

        Returns:
            preds (list) : 预测的类别（原始类别，非数字化类别）
        """
        assert isinstance(test_data, list)
        test_data, _ = self.fe.build_test(test_data)
        test_tfidf = self.fe.transform_tfidf(test_data)
        preds = self.predicate_pure(test_tfidf)
        return [self.fe.cates_inverse[i] for i in preds]

    def train_pure(self, train_data, train_label):
        """构建模型"""
        self.model.fit(train_data, train_label)

    def predicate_pure(self, test_data):
        """预测"""
        preds = self.model.predict(test_data)
        return numpy.array(preds, numpy.int32)

    def predicate_pure_proba(self, test_data):
        """预测"""
        preds = self.model.predict_proba(test_data)
        return preds

    def validate_pure(self, test_data, test_label, raw_docs=None, detail=False):
        """验证"""
        preds = self.predicate_pure(test_data)
        confusion_m = numpy.zeros((len(self.fe.cates), len(self.fe.cates)))
        for idx, x in enumerate(preds):
            confusion_m[test_label[idx], x] += 1
        pprint.pprint(confusion_m)
        for cate, idx in self.fe.cates.items():
            print("{}|| Precision: [{:.2%}], Recall: [{:.2%}]".format(
                cate,
                confusion_m[idx][idx]/sum(confusion_m[:, idx]),
                confusion_m[idx][idx]/sum(confusion_m[idx, :])))
        if detail and raw_docs is not None:
            for idx, x in enumerate(preds):
                if x != test_label[idx]:
                    # 输出预测错误的样本
                    print("pred:[{}], raw:[{}]".format(self.fe.cates_inverse[x], raw_docs[idx])) 

    def validate(self, test_file, fmt='json', split='__label__', detail=False):
        """载入测试数据进行验证
        """
        test_docs = [line.strip() for line in open(test_file, encoding="utf-8").readlines()]
        test_data, test_label = self.fe.build_test(test_docs, fmt=fmt, split=split)
        print(">>>SHAPE:", test_data.shape)
        idx = self.fe.check_nan(test_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before testing: ", idx)
            test_data, test_label = self.fe.handling_empty_line(test_data, label=test_label, idxs=idx)
            new_test_docs = [test_docs[i] for i in range(len(test_docs)) if i not in set(idx[0])]
        else:
            new_test_docs = [i for i in test_docs]
        test_tfidf = self.fe.transform_tfidf(test_data)
        self.validate_pure(test_tfidf, test_label, new_test_docs, detail)

    def output_close(self, test_file, fmt='json', split='__label__'):
        """输出测试结果很接近的数据
        """
        test_docs = [line.strip() for line in open(test_file, encoding="utf-8").readlines()]
        test_data, test_label = self.fe.build_test(test_docs, fmt=fmt, split=split)
        idx = self.fe.check_nan(test_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before testing: ", idx)
            test_data, test_label = self.fe.handling_empty_line(test_data, label=test_label, idxs=idx)
            # 获得新的原始数据
            new_test_docs = [test_docs[i] for i in range(len(test_docs)) if i not in set(idx[0])]
        else:
            new_test_docs = [i for i in test_docs]
        test_tfidf = self.fe.transform_tfidf(test_data)
        preds = self.predicate_pure_proba(test_tfidf)
        for idx in range(preds.shape[0]):
            pred_c = numpy.argmax(preds)
            pred = sorted(preds[idx], reverse=True)
            if (pred[0] - pred[1]) / pred[1] < .1:
                # 不好区分
                # print(json.dumps(raw_data[idx], ensure_ascii=False))
                print("cate:> {}".format(self.fe.cates_inverse[pred_c]), new_test_docs[idx])
                print("proba:> ||raw> {}".format(pred))


if __name__ == "__main__":
    mymodel = SVMModel()  # 实例化
    mymodel.train("../data/train_data.csv", fmt='csv', split='__label__')  # 训练
    mymodel.save_model("../model/proj_doc_svm.model", "../model/proj_doc_feature_svm.model")  # 保存模型
    print("训练集作为测试集的结果如下:")
    mymodel.validate("../data/test_data.csv", fmt='csv', split='__label__')  # 测试
