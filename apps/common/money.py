# -*- coding: utf-8 -*-
"""money.py
~~~~~~~~~~~
本模块提供数字转化相关工具函数
"""
import re
import math


class Numbers:
    """人民币相关功能
    """
    def __init__(self):
        """初始化"""
        self.number = {
            '〇' : 0, '一' : 1, '二' : 2, '三' : 3, '四' : 4, '五' : 5, '六' : 6, '七' : 7, '八' : 8, '九' : 9, '零' : 0,
            '壹' : 1, '贰' : 2, '叁' : 3, '肆' : 4, '伍' : 5, '陆' : 6, '柒' : 7, '捌' : 8, '玖' : 9, '貮' : 2, '两' : 2,
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
        }
        self.unit = {
            '十' : 10, '拾' : 10, '百' : 100, '佰' : 100, '千' : 1000, '仟' : 1000,
            '万' : 10000, '萬' : 10000, '亿' : 100000000, '億' : 100000000
        }
        self.money_numbers = r"^(\d+)(,\d\d\d)*(\.\d+)?$"
        self.money_numbers_units = r"^(?P<digit>(\d+)(,\d\d\d)*(\.\d+)?)(?P<unit>[万亿十百千]+)$"
        self.ch_denom = r"^[一二三四五六七八九十百千]+分之[一二三四五六七八九十百]+$"

    def number_with_area(self, area):
        """将面积统计表示

        Args:
            area (:str) : 带单位的面积字符串

        Returns:
            out (:int or :float) : 统一单位为平米的整形/浮点型面积
        """
        if area.strip():
            narea = -1
            if area.endswith("亩"):
                area = area[:-1]
                narea = self.change_to_digits(area)
                if narea > 0:
                    narea *= 667
                return narea
            for unit in ["平方米", "平米", "m2"]:
                if area.endswith(unit):
                    area = area[:-len(unit)]
                    narea = self.change_to_digits(area)
                    return narea
            narea = self.change_to_digits(area)
            return narea
        return -1

    def number_with_default(self, cate):
        """数字统一表示的默认方法

        Args:
            area (:str) : 带单位的种类数量字符串

        Returns:
            out (:int or :float) : 统一单位后的整形/浮点型种类数量
        """
        if cate.strip():
            narea = self.change_to_digits(cate)
            return narea
        return -1

    def number_with_unit(self, money, unit="万"):
        """将数字字符串转化为为带单位的数字

        >>> money = Numbers()
        >>> print(money.number_with_unit('伍仟陆佰', "百"))
        56.0
        >>> print(money.number_with_unit('九千八百七十', '千'))
        9.87
        >>> print(money.number_with_unit('九千八百七十', '万'))
        0.99
        >>> print(money.number_with_unit('一亿三千八百二十五万零肆佰玖拾捌', '万'))
        13825.05

        Args:
            money (:str) : 字符串金额

        Returns:
            out (:int or float) : 转化单位后的金额
        """
        if money.strip():
            if unit not in self.unit:
                # 如果unit不存在，默认为1
                nmoney = self.change_to_digits(money)
                return nmoney
            nmoney = self.change_to_digits(money)
            chmoney = nmoney / self.unit[unit]
            return round(chmoney, 2)
        return -1

    def change_to_digits(self, money, round_digits=2):
        """将数字字符串转化为数字

        >>> money = Numbers()
        >>> print(money.change_to_digits('三百五十一'))
        351
        >>> print(money.change_to_digits('三仟零五十二'))
        3052
        >>> print(money.change_to_digits('肆佰玖拾捌'))
        498
        >>> print(money.change_to_digits('伍仟陆佰'))
        5600
        >>> print(money.change_to_digits('九千八百七十'))
        9870
        >>> print(money.change_to_digits('一亿三千八百二十五万零肆佰玖拾捌'))
        138250498
        >>> print(money.change_to_digits('三万零陆佰二十七亿三千八百二十五万零肆佰玖拾捌'))
        3062738250498
        >>> print(money.change_to_digits('五百萬'))
        5000000
        >>> print(money.change_to_digits('五百萬零一'))
        5000001
        >>> print(money.change_to_digits('五百萬零一百二十'))
        5000120
        >>> print(money.change_to_digits('五百萬零一百二十八'))
        5000128
        >>> print(money.change_to_digits('一'))
        1
        >>> print(money.change_to_digits('一十'))
        10
        >>> print(money.change_to_digits('十一'))
        11
        >>> print(money.change_to_digits('肆十一'))
        41
        >>> print(money.change_to_digits('一十五'))
        15
        >>> print(money.change_to_digits('贰百'))
        200
        >>> print(money.change_to_digits('一百一十一'))
        111
        >>> print(money.change_to_digits('一百零一'))
        101
        >>> print(money.change_to_digits('一百一十'))
        110
        >>> print(money.change_to_digits('一千五百一十'))
        1510
        >>> print(money.change_to_digits('一千零一十'))
        1010
        >>> print(money.change_to_digits('十一万'))
        110000
        >>> print(money.change_to_digits('一十一万'))
        110000
        >>> print(money.change_to_digits('128938.123'))
        128938.12
        >>> print(money.change_to_digits('12893123'))
        12893123
        >>> print(money.change_to_digits('1000万'))
        10000000
        >>> print(money.change_to_digits('1,000'))
        1000
        >>> print(money.change_to_digits('1000'))
        1000
        >>> print(money.change_to_digits('1,000.15'))
        1000.15
        >>> print(money.change_to_digits('1,000.15万'))
        10001500
        >>> print(money.change_to_digits('1,234,000.567万'))
        12340005670
        >>> print(money.change_to_digits('2.33万'))
        23300
        >>> print(money.change_to_digits('15%'))
        0.15
        >>> print(money.change_to_digits('16.13%'))
        0.16
        >>> print(money.change_to_digits('百分之三十八'))
        0.38
        >>> print(money.change_to_digits('百分之一百六十二'))
        1.62
        >>> print(money.change_to_digits('三分之一'))
        0.33
        >>> print(money.change_to_digits('二十五分之一'))
        0.04
        >>> print(money.change_to_digits(''))
        -1

        Args:
            money (:str) : 金额

        Returns:
            out (:tuple) : 数字金额以及标志（标志数字合法与否）,-1为异常返回值
        """
        if not money.strip():
            return -1
        rmoney = ""
        if re.search(self.money_numbers, money) is not None:
            # 阿拉伯数字，只需将`,`去掉即可
            rmoney = float(money.replace(",", ""))
        else:
            # 阿拉伯数字加单位
            mobj = re.search(self.money_numbers_units, money)
            if mobj is not None:
                dmoney = mobj.group("digit")
                unit = mobj.group("unit")
                dmoney = dmoney.replace(",", "")
                dunit = 1
                for ch in unit:
                    dunit *= self.unit[ch]
                rmoney = float(dmoney) * dunit
            else:
                # 百分号
                if money.find("%") != -1:
                    rmoney = float(money[:-1]) / 100
                else:
                    # *分之*
                    if re.search(self.ch_denom, money) is not None:
                        dnom, nom = money.split("分之")
                        nnom = self.change_regular(nom)
                        ndnom = self.change_regular(dnom)
                        if ndnom == 0:
                            rmoney = 0
                        else:
                            rmoney = nnom / ndnom
                    else:
                        rmoney = self.change_regular(money)
        if rmoney:
            if int(rmoney) == float(rmoney):
                return int(rmoney)
            else:
                return round(rmoney, round_digits)
        return -1

    def change_regular(self, money):
        """将标准的数字字符串转化为浮点型

        Args:
            money () :

        Returns:
            out () :
        """
        total = 0
        para_total = 0
        last_unit = 0
        tmp = 0
        for ch in money:
            if ch in self.number:
                tmp = self.number[ch]
            elif ch in self.unit:
                if self.unit[ch] == 10000 or self.unit[ch] == 100000000:
                    # `万`和`亿`两个单位需要特殊处理
                    if last_unit and last_unit >= self.unit[ch]:
                        if para_total:
                            total += (para_total + tmp) * self.unit[ch]
                        else:
                            total += tmp * self.unit[ch]
                    else:
                        if para_total:
                            total = (total + para_total + tmp) * self.unit[ch]
                        else:
                            total += tmp * self.unit[ch]
                    para_total = 0
                    last_unit = self.unit[ch]
                else:
                    if tmp:
                        para_total += tmp * self.unit[ch]
                    else:
                        para_total = self.unit[ch]
                tmp = 0
        if tmp != 0:
            total += tmp
        if para_total != 0:
            total += para_total
        return total
