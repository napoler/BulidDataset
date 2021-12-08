import re
import unicodedata
from transformers import BertTokenizerFast

import unicodedata


class AutoClear:
    """

    用于清理空格等等数据

    """

    def __init__(self, seg=None, tokenizer=None):

        pass

    def filterPunctuation(self, x):
        """[summary]

        中文标点转换成英文
        Args:
            x ([type]): [description]
        Returns:
            [type]: [description]
        """
        x = re.sub(r'[‘’]', "'", x)
        x = re.sub(r'[“”]', '"', x)
        x = re.sub(r'[…]', '...', x)
        x = re.sub(r'[—]', '-', x)
        # x = re.sub(r'[]', '℃', x)
        x = re.sub(r"&nbsp", "", x)

        E_pun = u',.!?[]()<>"\''
        C_pun = u'，。！？【】（）《》“‘'

        table = {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
        x = x.translate(table)
        x = x.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        return x

    def clearText(self, text):
        """
        清理文本中的回车等
        """
        text_or = text.lower()
        # 中文标点转换英文
        # text_or=unicodedata.normalize('NFKD',text_or)
        text_or = self.filterPunctuation(text_or)
        # 使用tab替换空格
        text = text_or.replace(" ", "ر").replace(
            "\t", "س").replace("\n", "ة").replace("\r", "ت")
        return text

    def clearTextDec(self, seg_list):
        """
        修正词语中的特殊符号
        返回为分词后空格分割
        """
        newtext = " ".join(seg_list)

        newtext = newtext.replace("ر", "[PAD]").replace("س", "[PAD]")
        newtext = newtext.replace("ة", "[SEP]").replace("ت", "[SEP]")
        return newtext.split(" ")

