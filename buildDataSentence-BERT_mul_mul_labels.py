# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
自动构建数据集 预处理使用

Sentence-BERT模式数据集
数据参考示例
dataDemo/Sentence-BERT.csv

"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import TensorDataset, random_split
from tqdm.auto import tqdm

from config import *
# 输出目录
from tkitDatasetEx import NpEncoder

path = "out"
MAX_LENGTH = 64
# /home/terry/PycharmProjects/auto_Icd_model/data/icd_data/out/data_mul_mini.csv

print("""
Sentence-BERT模式数据集
多标签任务，单个标签任务过于简单，增加复杂度


数据参考示例
dataDemo/Sentence-BERT.csv

支持传入参数 类似

> python buildDataSentence-BERT.py out/train.csv


""")

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

if dataFile:
    print("dataFile", dataFile)
    df = pd.read_csv(dataFile)
    df.drop_duplicates()
    # remove null data
    # 删除表中含有任何NaN的行
    df.dropna(axis=0, how='all')

dataA = df["sent1"].squeeze().str.lower().astype(str).values.tolist()
dataB = df["sent2"].squeeze().str.lower().astype(str).values.tolist()
dataLabel = df["label"].squeeze().values.tolist()
# print(dataLabel)
# 获取label标签
le.fit(dataLabel)
print(le.classes_)

labels = list(le.classes_)
# 获取标签格式数据
tgt = le.transform(dataLabel)

print("labels", labels)
print("labels len：", len(labels))

df["sent1"] = dataA
df["sent2"] = dataB
df["label"] = tgt

print(df)


def auto_data(df, idx):
    for index in range(2):
        max_num = df.shape[0]
        num_list = list(range(max_num))
        idxs = np.random.randint(0, max_num, size=index)
        # print([num_list[i] for i in idxs])
        r_list = [num_list[i] for i in idxs]
        r_list.append(idx)
        # print(r_list)
        item = {
            "sent1": [],
            "sent2": [],
            "label": []
        }
        for i in r_list:
            # print(df.loc[i])
            row = df.loc[i]
            # print(row)
            item['sent1'].append(row['sent1'])
            item['sent2'].append(row['sent2'])
            item['label'].append(row['label'])
        # print(item)
        items = {"sent1": "", "sent2": "", "label": [], 'sent1': "".join(item['sent1']),
                 'sent2': "".join(item['sent2'])}
        label = item['label'] + [1599] * 3
        items['label'] = label[:3]
        yield items
    # return items


dataA = []
dataB = []
tgt = []

with open(path + "/data.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=["sent1", "sent2", "label"])
    w.writeheader()
    for idx, data in tqdm(df.iterrows()):
        # print(data)
        for items in auto_data(df, idx):
            # print(items)
            # df.loc['1']
            # dataA = dataA + items['sent1']
            # dataB = dataB + items['sent2']
            # tgt = tgt + items['label']
            w.writerow(items)
            dataA.append(items['sent1'])
            dataB.append(items['sent2'])
            tgt.append(items['label'])

        # if idx > 10:
        #     break
        pass

df = pd.read_csv(path + "/data.csv")
print(df)
# #
# # exit()
#
# dataA = df["sent1"].squeeze().str.lower().astype(str).values.tolist()
# dataB = df["sent2"].squeeze().str.lower().astype(str).values.tolist()
# tgt = df["label"]
# .squeeze().values.astype(list).tolist()
#
# tgt = torch.Tensor(tgt)

print("inputsA", len(dataA))
print("dataB", len(dataB))

inputsA = tokenizer(dataA, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
inputsB = tokenizer(dataB, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
# inputsLabels = torch.Tensor(tgt)
inputsLabels = torch.Tensor(tgt)
# print(inputsA['input_ids'].size())
traindataset = TensorDataset(inputsA['input_ids'], inputsA['attention_mask'],
                             inputsB['input_ids'], inputsB['attention_mask'],
                             inputsLabels
                             )

fullLen = len(traindataset)
trainLen = int(fullLen * 0.7)
valLen = int(fullLen * 0.15)
testLen = fullLen - trainLen - valLen

train, val, test = random_split(traindataset, [trainLen, valLen, testLen])

try:
    os.makedirs(path)
except:
    pass
# 保存标签信息
# print("labels",type(labels))
with open(path + "/labels.json", 'w', encoding="utf-8") as f:
    # l=json.dumps(labels)
    json.dump(labels, f, ensure_ascii=False, cls=NpEncoder)

torch.save(train, path + "/train.pkt")
torch.save(val, path + "/val.pkt")
torch.save(test, path + "/test.pkt")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')

    pass
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
