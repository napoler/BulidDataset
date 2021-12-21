# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
自动构建数据集 预处理使用
buildDataBertSequencePair模式数据集
数据参考示例
dataDemo/Sentence-BERT.csv




"""
import json
import os
import sys

import numpy
import torch
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn import preprocessing
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizerFast

from tkitDatasetEx.fun import NpEncoder

tokenizer = BertTokenizerFast.from_pretrained("tokenizer", do_basic_tokenize=True)
print("""
自动构建数据集 预处理使用
BIEO模式数据集
数据参考示例
dataDemo/label-studio-ner.json


""")
# 输出目录
path = "out"
MAX_LENGTH = 128

try:
    os.makedirs(path)
except:
    pass
if len(sys.argv) > 1:
    dataFile = sys.argv[1]
    MAX_LENGTH = sys.argv[2]
else:
    dataFile = input("数据集地址：")

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
    text = item['data']['text']
    # tags = ["O"] * len(text)
    tags = ["O"] * MAX_LENGTH
    for it in item["annotations"]:
        # print(it['result'])
        for i, iit in enumerate(it['result']):
            # print(i, iit)
            if iit['type'] == "labels":
                for iii in range(iit['value']['start'], iit['value']['end']):
                    if iii == iit['value']['start']:
                        tags[iii + 1] = "B-" + iit['value']['labels'][0]
                    elif iii == iit['value']['end'] - 1:
                        tags[iii + 1] = "E-" + iit['value']['labels'][0]
                    else:
                        tags[iii + 1] = "I-" + iit['value']['labels'][0]

    # print(tags)
    words = list(text)
    for i, (w, t) in enumerate(zip(words, tags)):
        # print(w,t)
        if w in [" ", "\t"]:
            words[i] = "[PAD]"
        elif w in ["\n", "\r"]:
            words[i] = "[SEP]"

    return words, tags


datas = {"labels": [], "text": [], "tags": [], "tags_ids": [], "words": []}
if dataFile:
    with open(dataFile, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i, it in enumerate(data):
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
