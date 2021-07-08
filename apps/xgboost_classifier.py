# -*- coding: utf-8 -*-
"""xgboost_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本模块实现xgboost分类器。

:copyright: (c) 2019 by Zhichao Xia
:modified: 2021-07-08

:TODO:
"""
import numpy
import xgboost
import pprint
import re
from apps.common.feature_engineering import FeatureEngineering


class MyEngineering(FeatureEngineering):
    def remove_stop_words(self, item):
        """去掉停用词"""
        if not isinstance(item, str):
            return False
        #if len(item) < 2:
        #    return False
        if re.search("[\u4e00-\u9fa5]", item) is None:
            return False
        return True


class XGBoostModel:
    """基于XGboost的分类模型
    """
    def __init__(self, model_file="", feat_file="", stop_file=""):
        """"""
        if model_file and feat_file:
            self.load_model(model_file, feat_file)
        else:
            if stop_file:
                self.fe = MyEngineering(stop_file=stop_file)
            else:
                self.fe = MyEngineering()
            self.model = None

    def save_model(self, model_path, feat_path):
        """保存模型"""
        assert self.model is not None
        self.model.save_model(model_path)
        self.fe.save_model(feat_path)

    def load_model(self, model_path, feat_path):
        """载入模型"""
        self.fe = FeatureEngineering(model_path=feat_path)
        self.model = xgboost.Booster(model_file=model_path)
        assert self.model is not None

    def train(self, train_file, valid_file=None, fmt='json', split='__label__', tfidf=True, binary=True):
        """从原始数据构建模型
        """
        train_data, train_label = self.fe.build_train(train_file, fmt=fmt, split=split)
        idx = self.fe.check_nan(train_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before training: ", idx)
            train_data, train_label = self.fe.handling_empty_line(train_data, label=train_label, idxs=idx)
        train_tfidf = self.fe.transform_tfidf(train_data)
        if valid_file is not None:
            valid_data, valid_label = self.fe.build_test(valid_file, fmt=fmt, split=split)
            idx = self.fe.check_nan(valid_data)
            if idx[0].any():
                print("Empty line found in data, PLEASE remove these line before validating: ", idx)
                valid_data, valid_label = self.fe.handling_empty_line(valid_data, label=valid_label, idxs=idx)
            valid_tfidf = self.fe.transform_tfidf(valid_data)
        else:
            valid_label = None
            valid_tfidf = None
        if tfidf:
            self.train_pure(train_tfidf, train_label, valid_tfidf, valid_label, binary=binary)
        else:
            self.train_pure(train_data, train_label, valid_data, valid_label, binary=binary)
        #self.train_pure(train_data, train_label, valid_data, valid_label)

    def predicate(self, test_data):
        """预测原始数据"""
        assert isinstance(test_data, list)
        test_data, _ = self.fe.build_test(test_data)
        idx = self.fe.check_nan(test_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before validating: ", idx)
            test_data, _ = self.fe.handling_empty_line(test_data, idxs=idx)
        test_tfidf = self.fe.transform_tfidf(test_data)
        dtest = xgboost.DMatrix(test_tfidf)
        preds = self.predicate_pure(dtest)
        return [self.fe.cates_inverse[i] for i in preds]

    def train_pure(self, train_data, train_label, valid_data=None, valid_label=None, binary=True):
        """构建模型"""
        if binary:
            param_model = {
                'learning_rate': 0.5,  # 学习率，默认0.1
                'n_estimators': 500,  # 决策树的数量，默认为100
                'max_depth': 5,  # 给定树的深度，默认为3
                'silent': 0,  # silent=0时，输出中间过程（默认），silent=1时，不输出中间过程
                'objective': 'binary:logistic',  # 目标函数，回归任务[reg:linear(默认)、reg:logistic]；
                                                 # 二分类[binary:logistic 概率、binary:logitraw 类别]；
                                                 # 多分类[multi:softmax num_class=n 返回类别、multi:softprob num_class=n 返回概率]
                'nthread': 1,  # 使用线程数nthread=-1时，使用全部CPU进行并行运算（默认）nthread=1时，使用1个CPU进行运算。
            }
            self.model = xgboost.XGBClassifier(**param_model)
            if valid_data is not None:
                # early_stopping_rounds：当logloss在10轮迭代之内，都没有提升的话，就stop。
                # eval_metric：rmse：均方根误差|||mae：平方绝对值误差|||logloss：负对数似然|||error：二分类错误率|||map: 平均正确率
                # merror：多分类错误率|||mlogloss：多分类log损失|||auc：曲线下的面积|||ndcg: Normalized Discounted Cumulative Gain
                #  eval_set，为了early stopping，如果出现loss在训练集上下降但在验证集上上升，说明出现过拟合迹象了，可以提前终止模型训练。
                self.model.fit(train_data, train_label, early_stopping_rounds=10, eval_metric="logloss",
                               eval_set=[(train_data, train_label), (valid_data, valid_label)])
            else:
                self.model.fit(train_data, train_label, early_stopping_rounds=10, eval_metric="logloss",
                               eval_set=[(train_data, train_label)])
        else:
            param_model = {
                'learning_rate': 0.7,
                'n_estimators': 500,
                'max_depth': 3,
                'silent': 0,
                'objective': 'multi:softmax',
                'num_class': numpy.unique(train_label).shape[0],  # 多分类类别数
                'nthread': 8,
            }
            self.model = xgboost.XGBClassifier(**param_model)
            if valid_data is not None:
                self.model.fit(train_data, train_label, early_stopping_rounds=10, eval_metric="merror",
                               eval_set=[(train_data, train_label), (valid_data, valid_label)])
            else:
                self.model.fit(train_data, train_label, early_stopping_rounds=10, eval_metric="merror",
                               eval_set=[(train_data, train_label)])

    def predicate_pure(self, test_data):
        """预测"""
        preds = self.model.predict(test_data)
        return numpy.array(preds, numpy.int32)

    def predicate_proba(self, test_data):
        """预测"""
        preds = self.model.predict_proba(test_data)
        return preds

    def output_close(self, test_file, fmt='json', split='__label__', tfidf=True):
        """预测"""
        test_docs = [line.strip() for line in open(test_file).readlines()]
        test_data, test_label = self.fe.build_test(test_docs, fmt=fmt, split=split)
        idx = self.fe.check_nan(test_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before testing: ", idx)
            test_data, test_label = self.fe.handling_empty_line(test_data, label=test_label, idxs=idx)
            # 获得新的原始数据
            new_test_docs = [test_docs[i] for i in range(len(test_docs)) if i not in set(idx[0])]
        else:
            new_test_docs = [i for i in test_docs]
        if tfidf:
            test_tfidf = self.fe.transform_tfidf(test_data)
            self.validate_pure(test_tfidf, test_label, new_test_docs)
        else:
            self.validate_pure(test_data, test_label, new_test_docs)
        preds = self.model.predict_proba(test_data)
        for idx in range(preds.shape[0]):
            pred_c = numpy.argmax(preds)
            pred = sorted(preds[idx], reverse=True)
            if (pred[0] - pred[1]) / pred[1] < .1:
                print("pred> {}||raw> {}".format(pred, new_test_docs[idx]))
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

    def validate(self, test_file, fmt='json', split='__label__', tfidf=True, detail=False):
        """载入测试数据进行验证
        """
        test_docs = [line.strip() for line in open(test_file, encoding='utf-8').readlines()]
        test_data, test_label = self.fe.build_test(test_docs, fmt=fmt, split=split)
        idx = self.fe.check_nan(test_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before testing: ", idx)
            test_data, test_label = self.fe.handling_empty_line(test_data, label=test_label, idxs=idx)
            # 获得新的原始数据
            new_test_docs = [test_docs[i] for i in range(len(test_docs)) if i not in set(idx[0])]
        else:
            new_test_docs = [i for i in test_docs]
        if tfidf:
            test_tfidf = self.fe.transform_tfidf(test_data)
            self.validate_pure(test_tfidf, test_label, new_test_docs, detail)
        else:
            self.validate_pure(test_data, test_label, new_test_docs, detail)


if __name__ == "__main__":
    mymodel = XGBoostModel(stop_file="../data/chineseStopWords.txt")
    mymodel.train(train_file="../data/train_data.csv", valid_file="../data/train_data.csv", binary=False, tfidf=False, fmt='csv', split='__label__')
    mymodel.validate("../data/test_data.csv", tfidf=False, fmt='csv', split='__label__')
    #mymodel.output_close("/Users/hongziki/tmp/requirement_test.txt", fmt='csv', split=",", tfidf=True)
    #mymodel.validate("/Users/hongziki/tmp/requirement_test.txt", fmt='csv', split=",", tfidf=False)
    #mymodel.save_model("./apps/model/proj_doc_xgboost.model", "./apps/model/proj_doc_feature.model")
    # with open("proj_doc_categorization.vocab", "w") as fobj:
    #     for feat, score in sorted(mymodel.model.get_booster().get_fscore().items(), key=lambda x: x[1], reverse=True):
    #         print(mymodel.fe.vocabulary_inverse[int(feat[1:])], score)
    #         fobj.write("{}\n".format(mymodel.fe.vocabulary_inverse[int(feat[1:])]))
