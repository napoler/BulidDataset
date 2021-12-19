# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
bio数据集

"""
# from transformers import BertTokenizerFast
import json
import os

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn import preprocessing

from AutoClear import AutoClear
from fun import NpEncoder


class BIO:
    def __init__(self, type="BIO", out_dir="out", clear=True, tokenizer=None):
        """


        :param type:
        :param out_dir:
        """
        self.type = type
        self.entity_dict = {}
        self.out_dir = out_dir
        self.clear = clear
        self.items = {"words": [], "tags": []}
        self.le = preprocessing.LabelEncoder()
        try:
            os.mkdir(self.out_dir)
        except:
            pass
        # if tokenizer
        self.apos = AutoClear()
        pass

    def encode(self, items=[]):
        """
        items=[{"text": "预测1：[CLS] <王陵 魅影 > 是 连载 于 17 k小说网 的 网部 玄墓小险类 小说,作者 是 皇甫 龙悦 [SEP]",
            "result": [
          {
            "value": {
              "start": 11,
              "end": 16,
              "text": "王陵 魅影",
              "labels": [
                "实体"
              ]
            },
            "id": "Xu4PLCCJAJ",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 25,
              "end": 33,
              "text": " 17 k小说网",
              "labels": [
                "实体"
              ]
            },
            "id": "4ArJ7bIX9e",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 21,
              "end": 25,
              "text": "连载 于",
              "labels": [
                "关系"
              ]
            },
            "id": "EaWMvn9EOn",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "from_id": "4ArJ7bIX9e",
            "to_id": "EaWMvn9EOn",
            "type": "relation",
            "direction": "right",
            "labels": []
          },
          {
            "from_id": "EaWMvn9EOn",
            "to_id": "Xu4PLCCJAJ",
            "type": "relation",
            "direction": "right",
            "labels": []
          }
        ],
        }]


        :return:
        """
        out_items = {"words": [], "tags": [], "tags_ids": []}
        for item in items:
            if self.clear:
                # self.apos
                text = item['text']
                text = self.apos.clearText(text)

            else:
                text = item['text']
            tags = ["O"] * len(text)
            # print(tags)
            for i, iit in enumerate(item['result']):
                print(i, iit)
                if iit['type'] == "labels":
                    try:
                        self.entity_dict[iit['value']['labels'][0]] += 1
                    except:
                        self.entity_dict[iit['value']['labels'][0]] = 1
                    for iii in range(iit['value']['start'], iit['value']['end']):
                        if iii == iit['value']['start']:
                            tags[iii + 1] = "B-" + iit['value']['labels'][0]
                        elif iii == iit['value']['end'] - 1:
                            tags[iii + 1] = "E-" + iit['value']['labels'][0]
                        else:
                            tags[iii + 1] = "I-" + iit['value']['labels'][0]
            # print(tags)
            WordList = self.apos.clearTextDec(list(text))
            # print(WordList)
            # one={"wordlist":WordList,"tags":tags}
            # print(self.entity_dict)
            # return one
            out_items['words'].append(WordList)
            out_items['tags'].append(tags)
            # out_items.append(one)
        # print(one)
        # pass

        self.items = out_items
        return out_items

    def bulid_labels(self):

        tags = []
        for item in self.items['tags']:
            tags.extend(item)
        # 获取label标签
        # print(tags)
        self.le.fit(tags)
        # print(le.classes_)
        self.labels = list(self.le.classes_)

    def fit(self):

        for i, item in enumerate(self.items['tags']):
            # tags.extend(item)
            tgt = self.le.transform(item)
            self.items['tags_ids'].append(list(tgt))
            # datas['text'].append(" ".join(words))
        pass

    def save(self):
        with open(os.path.join(self.out_dir, "entity_dict.json"), 'w', encoding="utf-8") as f:
            # json.dump(datas, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            json.dump(self.entity_dict, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        with open(os.path.join(self.out_dir, "labels.json"), 'w', encoding="utf-8") as f:
            # json.dump(datas, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            json.dump(self.labels, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        with open(os.path.join(self.out_dir, "datas.json"), 'w', encoding="utf-8") as f:
            # json.dump(datas, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            json.dump(self.items, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        pass

    def auto(self, items):
        self.encode(items)
        self.bulid_labels()
        self.fit()
        self.save()
        return self.labels, self.items


if __name__ == '__main__':
    bio = BIO()
    item = {"text": "预测1：[CLS] <王陵 魅影 > 是 连载 于 17 k小说网 的 网部 玄墓小险类 小说,作者 是 皇甫 龙悦 [SEP]",
            "result": [
                {
                    "value": {
                        "start": 11,
                        "end": 16,
                        "text": "王陵 魅影",
                        "labels": [
                            "实体"
                        ]
                    },
                    "id": "Xu4PLCCJAJ",
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels"
                },
                {
                    "value": {
                        "start": 25,
                        "end": 33,
                        "text": " 17 k小说网",
                        "labels": [
                            "实体"
                        ]
                    },
                    "id": "4ArJ7bIX9e",
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels"
                },
                {
                    "value": {
                        "start": 21,
                        "end": 25,
                        "text": "连载 于",
                        "labels": [
                            "关系"
                        ]
                    },
                    "id": "EaWMvn9EOn",
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels"
                },
                {
                    "from_id": "4ArJ7bIX9e",
                    "to_id": "EaWMvn9EOn",
                    "type": "relation",
                    "direction": "right",
                    "labels": []
                },
                {
                    "from_id": "EaWMvn9EOn",
                    "to_id": "Xu4PLCCJAJ",
                    "type": "relation",
                    "direction": "right",
                    "labels": []
                }
            ],
            }
    out = bio.auto([item])
    print(out)
    pass
