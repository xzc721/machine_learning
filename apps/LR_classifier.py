# -*- coding: utf-8 -*-
"""LR_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本模块实现基于逻辑回归分类器。
- 代码利用sklearn的LogisticRegression进行训练
- 利用joblib完成机器学习模型保存和加载

:copyright: (c) 2019 by Zhichao Xia
:modified: 2021-06-22

"""
import numpy
import re
import pprint
import joblib
from sklearn.linear_model import LogisticRegression
from apps.common.utils import validat_requirements
from apps.common.feature_engineering import FeatureEngineering


class MyEngineering(FeatureEngineering):
    def remove_stop_words(self, item):
        """去掉停用词"""
        if not isinstance(item, str):
            return False
        if len(item) < 2:
            return False
        if re.search("[\u4e00-\u9fa5]", item) is None:
            return False
        return True


class LRModel:
    """基于逻辑回归的分类模型
    """
    def __init__(self, model_file="", feat_file=""):
        """"""
        if model_file and feat_file:
            self.load_model(model_file, feat_file)
        else:
            self.fe = MyEngineering()
            self.model = LogisticRegression(class_weight="balanced")
        # 创建一个申报条件汇总的初始列表
        self.requirements_list = []

    def save_model(self, model_path, feat_path):
        """保存模型"""
        assert self.model is not None
        joblib.dump(self.model, model_path)
        self.fe.save_model(feat_path)

    def load_model(self, model_path, feat_path):
        """载入模型"""
        self.fe = FeatureEngineering(model_path=feat_path)
        self.model = joblib.load(model_path)
        assert self.model is not None

    def train(self, train_file, fmt='csv', split=',', tfidf=True):
        """从原始数据构建模型
        """
        train_data, train_label = self.fe.build_train(train_file, fmt=fmt, split=split)
        idx = self.fe.check_nan(train_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before training: ", idx)
            train_data, train_label = self.fe.handling_empty_line(train_data, label=train_label, idxs=idx)
        train_tfidf = self.fe.transform_tfidf(train_data)
        self.train_pure(train_tfidf, train_label)

    def predicate(self, test_data):
        """预测原始数据"""
        assert isinstance(test_data, list)
        test_data, _ = self.fe.build_test(test_data, fmt='csv', split=None)
        idx = self.fe.check_nan(test_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before validating: ", idx)
            test_data, _ = self.fe.handling_empty_line(test_data, idxs=idx)
        test_tfidf = self.fe.transform_tfidf(test_data)
        preds = self.predicate_pure(test_tfidf)
        return [self.fe.cates_inverse[i] for i in preds]

    def train_pure(self, train_data, train_label):
        """构建模型"""
        self.model.fit(train_data, train_label)

    def predicate_pure(self, test_data):
        """预测分类结果"""
        preds = self.model.predict(test_data)
        return numpy.array(preds, numpy.int32)

    def predicate_pure_proba(self, test_data):
        """预测"""
        preds = self.model.predict_proba(test_data)
        return preds

    def validate_pure(self, test_data, test_label, raw_docs=None):
        """验证分类效果，输出混淆矩阵和PR值"""
        print("---------:", test_data.shape, test_label.shape, len(raw_docs))
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
        if raw_docs is not None:
            for idx, x in enumerate(preds):
                if x != test_label[idx]:
                    # 输出预测错误的样本
                    print("pred:[{}], raw:[{}]".format(self.fe.cates_inverse[x], raw_docs[idx]))

    def validate(self, test_file, fmt='csv', split=',', tfidf=True):
        """载入测试数据进行验证
        """
        test_docs = [line.strip() for line in open(test_file, encoding="utf-8").readlines()]
        test_data, test_label = self.fe.build_test(test_file, fmt=fmt, split=split)
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

    def output_close(self, test_file, fmt='csv', split=','):
        """输出测试结果很接近的数据
        """
        import json
        raw_data = [json.loads(l) for l in open(test_file).readlines()]
        raw_data1 = [item["title"]+item["content"] for item in raw_data]
        test_data, test_label = self.fe.build_test(raw_data1, fmt=fmt, split=split)
        test_tfidf = self.fe.transform_tfidf(test_data)
        preds = self.predicate_pure_proba(test_tfidf)
        for idx in range(preds.shape[0]):
            pred = sorted(preds[idx], reverse=True)
            if ((pred[0] - pred[1]) / pred[1] < .5):
                # 不好区分
                # print(json.dumps(raw_data[idx], ensure_ascii=False))
                print(raw_data[idx])

    def sum_requirements_extractor(self, content):
        """抽取公文中的申报条件"""
        # 对申报公文以行为单位进行分割
        seg_content = content.split("\n")
        test_data, _ = self.fe.build_test(seg_content, fmt="csv", split=None)
        # print(test_data)
        idx = self.fe.check_nan(test_data)
        if idx[0].any():
            print("Empty line found in data, PLEASE remove these line before testing: ", idx)
            test_data, _ = self.fe.handling_empty_line(test_data, idxs=idx)
            # 获得新的原始数据
            new_test_docs = [seg_content[i] for i in range(len(seg_content)) if i not in set(idx[0])]
            # print("新的预测文档为", new_test_docs)
        else:
            new_test_docs = [i for i in seg_content]
        test_tfidf = self.fe.transform_tfidf(test_data)
        # print("test_tfidf为", test_tfidf)
        # print("test_tfidf的类型是", type(test_tfidf))
        res = self.predicate_pure(test_tfidf)
        # print(res)
        ret = [new_test_docs[i] for i in range(res.shape[0]) if self.fe.cates_inverse[res[i]] == "1"]
        return "\n".join(ret)


if __name__ == "__main__":
    mymodel = LRModel()
    # 训练模型
    mymodel.train("../data/train_data.csv", fmt='csv', split='__label__')
    mymodel.save_model("../model/proj_doc_lr.model", "../model/proj_doc_feature.model")
    # validat_requirements("D:/all_train_20191024.xlsx", mymodel.sum_requirements_extractor)
    # 测试模型
    print("训练集作为测试集的结果如下:")
    mymodel.validate("../data/test_data.csv", fmt='csv', split='__label__')
    # # print("------------------------------------------------------")
    # print("测试集作为测试集的结果如下:")
    # mymodel.validate("D:/newest_requirement_test1017.txt")
    # res = mymodel.predicate(["（一）734hfghjhjgdshsdjgfshjjgjfgdhfsjsgjfs"])
    # print(type(res))
    # print("申报条件的结果是:", res)
    # requirements_1 = mymodel.sum_requirements_extractor("一、申报主体\r\n（一）在江苏省内注册，具有独立法人资格、健全的财务管理机构和财务管理制度的规模以上工业企业。\r\n（二）未列入统计口径规模以上工业企业的企业，如达到规模以上工业企业标准，且具有独立法人资格，可由企业提供情况说明及相关材料，由所在地县级以上（含县级）统计部门审核，省统计局复核后，如符合相关标准，可视同为规模以上工业企业申报。\r\n 2、企业技术改造投资项目备案通知书或核准批复复印件；")
    # print(requirements_1)
    # requirements_2 = mymodel.sum_requirements_extractor("djkhdkjfhjffhhf\r\n8765877676887687\r\n（一）在江苏省内注册，具有独立法人资格、健全的财务管理机构和财务管理制度的规模以上工业企业。\r\n（二）未列入统计口径规模以上工业企业的企业，如达到规模以上工业企业标准，且具有独立法人资格，可由企业提供情况说明及相关材料，由所在地县级以上（含县级）统计部门审核，省统计局复核后，如符合相关标准，可视同为规模以上工业企业申报。\r\n 2、企业技术改造投资项目备案通知书或核准批复复印件；\n")
    # print("申报的条件是", requirements_2)
    #mymodel.output_close("/Users/hongziki/tmp/project_shuf5000.json")
    # mymodel.save_model("./apps/model/proj_doc_lr.model", "./apps/model/proj_doc_feature.model")
    #with open("proj_doc_categorization.vocab", "w") as fobj:
    #    for feat, score in sorted(mymodel.model.get_score(fmap='', importance_type='gain').items(), key=lambda x: x[1], reverse=True):
    #        print(mymodel.fe.vocabulary_inverse[int(feat[1:])], score)
    #        fobj.write("{}\n".format(mymodel.fe.vocabulary_inverse[int(feat[1:])]))
