# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
自动构建数据集 预处理使用

label studio v0.9 data ner pre




"""
import json
import os
import sys

import numpy
import torch
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn import preprocessing
from torch.utils.data import random_split, TensorDataset
from transformers import BertTokenizerFast

from tkitDatasetEx.fun import NpEncoder
from tkitDatasetEx.AutoClear import AutoClear
from tkitDatasetEx.readData import readDir

tokenizer = BertTokenizerFast.from_pretrained("tokenizer", do_basic_tokenize=True, model_max_length=1000000, )

print("tokenizer", tokenizer)

# 初始化自动修正
apos = AutoClear(tokenizer=tokenizer)
print("""
自动构建数据集 预处理使用
BIEO模式数据集
数据参考示例
dataDemo/label-studiov09


""")
# 输出目录
path = "out"
MAX_LENGTH = 2048

try:
    os.makedirs(path)
except:
    pass
if len(sys.argv) > 1:
    dataFile = sys.argv[1]
    MAX_LENGTH = sys.argv[2]
else:
    dataFile = input("数据集 ：")

try:
    if MAX_LENGTH is None:
        MAX_LENGTH = input("数据最大长度：")
        MAX_LENGTH = int(MAX_LENGTH)
except:
    if MAX_LENGTH > 0:
        pass
    else:
        MAX_LENGTH = 512
MAX_LENGTH = int(MAX_LENGTH)

le = preprocessing.LabelEncoder()


def one(item):
    """
    处理单个标注
    基于BIEO模式


    :param item:
    :return:
    """
    # print("annotations",item["annotations"])
    # text=item["annotations"]["text"]
    # print(item.keys())
    # print(item['data']['text'])

    text = item['text']
    text = apos.clearText(text)
    # tags = ["O"] * len(text)
    tags = ["O"] * MAX_LENGTH
    # print(it['result'])
    for i, iit in enumerate(it['label']):
        # print(i, iit)

        for iii in range(iit['start'], iit['end']):
            if iii > MAX_LENGTH + 1 or iit['start'] > MAX_LENGTH or iit['end'] > MAX_LENGTH:
                continue
            # print(iii,iit)
            try:
                if iii == iit['start']:
                    tags[iii + 1] = "B-" + iit['labels'][0]
                elif iii == iit['end'] - 1:
                    tags[iii + 1] = "E-" + iit['labels'][0]
                else:
                    tags[iii + 1] = "I-" + iit['labels'][0]
            except:
                print(" error pass")
                pass
    # print(tags)
    words = list(text)
    words = apos.clearTextDec(words)
    return words, tags


datas = {"labels": [], "text": [], "tags": [], "tags_ids": [], "words": []}
if dataFile:
    for i, it in enumerate(readDir(dataFile)):
        # print(it)
        words, tags = one(it)
        datas['words'].append(words)
        datas['tags'].append(tags)
        # print(words,tags)
        # break

# print(datas)

tags = []
for item in datas['tags']:
    tags.extend(item)
# 获取label标签
# print(tags)
le.fit(tags)
print(le.classes_)
#
labels = list(le.classes_)
# # 获取标签格式数据

for item, words in zip(datas['tags'], datas['words']):
    # tags.extend(item)
    tgt = le.transform(item)
    datas['tags_ids'].append(tgt)
    datas['text'].append(" ".join(words))
#
print("labels", labels)
print("labels len：", len(labels))
datas['labels'] = labels

with open(os.path.join(path, "data.json"), 'w', encoding="utf-8") as f:
    json.dump(datas, f, ensure_ascii=False, indent=4, cls=NpEncoder)

#
# # print(tgt)
#
inputsA = tokenizer(datas['text'], return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
tgt = torch.Tensor(numpy.array(datas['tags_ids']))
#
# # print(inputsA)
traindataset = TensorDataset(inputsA['input_ids'], inputsA['token_type_ids'], inputsA['attention_mask'],
                             tgt
                             )

fullLen = len(traindataset)
trainLen = int(fullLen * 0.7)
valLen = int(fullLen * 0.15)
testLen = fullLen - trainLen - valLen

train, val, test = random_split(traindataset, [trainLen, valLen, testLen])

#
# 保存标签信息
# print("labels",type(labels))
with open(path + "/labels.json", 'w', encoding="utf-8") as f:
    # l=json.dumps(labels)
    json.dump(labels, f, ensure_ascii=False, cls=NpEncoder)
#
torch.save(train, path + "/train.pkt")
torch.save(val, path + "/val.pkt")
torch.save(test, path + "/test.pkt")

# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     # print_hi('PyCharm')
#
#     pass
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
