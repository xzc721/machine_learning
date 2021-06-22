"""lexicon.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本模块提供词法分析相关功能，具体包括：分词、词性标注和命名体识别。

:copyright: (c) 2021 by Zhichao Xia
:modified: 2021-06-18
"""
import jieba
import jieba.posseg


def cut_words(input_str):
    """分词"""
    return " ".join(jieba.cut(input_str))


def cut_words_with_pos(input_str):
    """分词并过滤"""
    return " ".join([item for item, flag in list(jieba.posseg.cut(input_str)) if flag.startswith("v") or flag.startswith("n")])


def cut_words_list(input_str):
    """分词"""
    return list(jieba.cut(input_str, cut_all=True))


def cut_words_with_pos_list(input_str):
    """分词并过滤"""
    return list([item for item, flag in list(jieba.posseg.cut(input_str)) if flag.startswith("v") or flag.startswith("n")])
