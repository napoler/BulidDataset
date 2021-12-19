import re

from transformers import BertTokenizer


class AutoClear:
    """

    用于清理空格等等数据



    """

    def __init__(self, seg=None, tokenizer=None, do_seg=False):
        if seg == None and do_seg == True:
            import pkuseg
            self.seg = pkuseg.pkuseg(model_name='medicine')  # 程序会自动下载所对应的细领域模型

        elif do_seg == True:
            self.seg = seg

        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained(
                "uer/chinese_roberta_L-8_H-512", do_basic_tokenize=False)

        else:
            self.tokenizer = tokenizer

        pass
        self.unk_vocab = []

        self.vocab = list(self.tokenizer.vocab)

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
        会自动清理词典无法解析的字符为空格
        """
        text_or = text.lower()
        # 中文标点转换英文
        # text_or=unicodedata.normalize('NFKD',text_or)
        text_or = self.filterPunctuation(text_or)

        # 使用tab替换空格
        text = text_or.replace("\xa0", "ر").replace("\ufeff", "ر").replace("\u3000", "ر").replace(" ", "ر").replace(
            "\t", "س").replace("\n", "ة").replace("\r", "ت")
        words = list(text)
        # print(list(self.tokenizer.vocab))
        # 自动清理词典无法解析的字符为空格
        for i, w in enumerate(words):
            # nw = self.tokenizer.vocab.get(w)
            if w in self.unk_vocab:
                words[i] = "ر"
            elif w not in self.vocab:
                words[i] = "ر"
                self.unk_vocab.append(w)
            # print(w, nw)
            # if w == "\n":
            #     text_or = text_or.replace(w, "")
        # print(text)
        return "".join(words)

    def clearTextDec(self, seg_list):
        """
        修正词语中的特殊符号
        返回为分词后空格分割
        """
        newtext = " ".join(seg_list)

        newtext = newtext.replace("ر", "[PAD]").replace("س", "[PAD]")
        newtext = newtext.replace("ة", "[SEP]").replace("ت", "[SEP]")
        return newtext.split(" ")

