# -*- coding: utf-8 -*-
"""utils.py
~~~~~~~~~~
本模块提供相关工具函数
"""
import re
import json
import pandas


def check_project(title, content):
    """粗过滤项目类公文"""
    if not isinstance(title, str):
        return False
    if not isinstance(content, str):
        return False
    if len(content) < 300:
        return False
    if re.search("函|结果|职务|详查|任职|抽查|大会|就业|资格考试|交易会|研修班|交流会|举办|矿证|注册|竞赛|考评|特派员|规程|纲要|变更|考试|签订|调查|决算|转让|任免|不予|备案|下发|公报|年报|检查|总结|干部|赛区|党组|报名|验收|三公|退字|博览会|对接|绩效评价|退回|专场|领导小组|清理|经营|核查|招生|节目预告|参展|考察|普查|加强|领取|贯彻落实|监督|毕业生|结题|督查|同志|展览会|公务员|学习|登记|讲解|召开|宣传|发放|监管|答辩|规定|录取|规定|章程|通报|培训|报告|新闻|名单|研讨会|答记者问|条例|招聘|座谈|解读|公示|转发|印发|下达|下拨|取消|办法|细则|意见|建议|政策|评审|会议|命令|批复|决定|答复|推迟|调整|延期|撤销|洽谈", title) is not None:
        return False
    if re.search("通知", title) is None:
        return False
    return True


def load_json(json_file, cate_field='cate', cont_field='title,docTitle,docContent,content'):
    """载入json格式数据

    从json_file生成训练数据以及对应的类别标签。json_file可以是一个文件路径，也可以是一个字符串列表。
    当cate_filed为None时，生成不包含类别的数据。

    Args:
        json_file: 训练文件路径或者训练数据列表
        cate_field: dict中的类别key
        cont_field: dict中的数据内容key，字符串形式，多个key以逗号","分割

    Returns:
        (data, label): 文本数据以及对应的类别标签
    """
    data = []
    label = []
    if isinstance(json_file, str):
        with open(json_file, encoding='utf-8') as fobj:
            raw_data = [line.strip() for line in fobj.readlines() if line.strip()]
    elif isinstance(json_file, list):
        raw_data = [line.strip() for line in json_file]
    else:
        raise ValueError("Not supported train data format.")
    for line in raw_data:
        line = line.replace("'", '"')
        try:
            jsond = json.loads(line.strip())
        except:
            continue
        cont = ""
        for field in cont_field.split(","):
            tmp = jsond.get(field, "")
            if bool(tmp):
                if tmp.find("tle") != -1:
                    cont = "{}\n{}".format(cont, tmp*5)
                else:
                    cont = "{}\n{}".format(cont, tmp)
        if cont.strip():
            if cate_field is not None:
                cate = jsond.get(cate_field, "")
                if cate:
                    data.append(cont.strip())
                    label.append(cate)
            else:
                data.append(cont.strip())
    return data, label


def load_csv(csv_file, splitor=","):
    """载入csv格式的数据

    Args:
        csv_file: 数据文件路径或者包含数据的列表
        splitor: 如果包含类别，则有分隔符，默认为逗号","

    Returns:
        (data, label): 文本数据以及对应的类别标签
    """
    data = []
    label = []
    if isinstance(csv_file, str):
        with open(csv_file, encoding='utf-8') as fobj:
            raw_data = [line.strip() for line in fobj.readlines() if line.strip()]
    elif isinstance(csv_file, list):
        raw_data = [line.strip() for line in csv_file]
    else:
        raise ValueError("Not supported test data format.")
    for line in raw_data:
        if splitor is not None:
            if line.find(splitor) != -1:
                cont, cate = line.strip().split(splitor, 1)
                if cate and cont:
                    data.append(cont)
                    label.append(cate)
        else:
            data.append(line.strip())
    return data, label


def validat_requirements(excel_file, predicate=None, field="requirements"):
    """验证条件抽取准确率

    Args:
        excel_file (str) : 测试数据路径，excel文件
        predicate (callable) : 预测函数

    Returns:
        None
    """
    pd = pandas.read_excel(excel_file)
    totals = pd.shape[0]
    validates = 0
    full_right = 0
    ninety_right = 0
    eighty_right = 0
    six_right = 0
    for i in range(pd.shape[0]):
        if not pandas.isna(pd.iloc[i][field]):
            validates += 1
            # 正确的条件数据tmpset
            tmpset = set([re.sub("^[\u3000]+", "", item.strip()) for item in pd.iloc[i][field].split("\n")])
            tmpset = set(list(filter(lambda x: x, tmpset)))
            print("\n\n>>>>raw content=================\n", pd.iloc[i]["content"])
            # print(">>>>raw require----------repr-------\n", repr(pd.iloc[i][field]))
            print(">>>>raw require-----------------\n", pd.iloc[i][field])
            preds = predicate(pd.iloc[i]["content"])
            print(">>>>pred require-----------------\n", preds)
            predset = set([re.sub("^[\u3000]+", "", item.strip()) for item in preds.split("\n")])
            # 交集intersec
            intersec = tmpset.intersection(predset)
            print(">>>>OUTPUT:", len(intersec), len(tmpset), len(predset))
            if len(intersec) == len(tmpset):
                print(">>> all right")
                full_right += 1
            if len(tmpset) - len(intersec) == 1:
                print(">>> 条件数抽取相差1个")
                ninety_right += 1
            if len(tmpset) - len(intersec) == 2:
                print(">>> 条件数抽取相差2个")
                eighty_right += 1
            if len(tmpset) - len(intersec) >= 3:
                print(">>> 条件数抽取相差3个及以上")
                six_right += 1
    print("TOTAL:", validates)
    print("全部抽取正确: ", full_right)
    print("条件数抽取相差1个的数量", ninety_right)
    print("条件数抽取相差2个的数量 ",  eighty_right)
    print("条件数抽取相差大于等于3个的数量 ", six_right)



if __name__ == '__main__':
    validat_requirements("/Users/hongziki/tmp/all_train_real.xlsx")
