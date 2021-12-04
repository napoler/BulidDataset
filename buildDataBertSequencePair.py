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

import pandas as pd
import torch
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn import preprocessing
from torch.utils.data import random_split, TensorDataset

from config import *
from libs.fun import NpEncoder

print("""
自动构建数据集 预处理使用
buildDataBertSequencePair模式数据集
数据参考示例
dataDemo/Sentence-BERT.csv

""")
# 输出目录
path = "out"
if len(sys.argv) > 1:
    dataFile = sys.argv[1]
    MAX_LENGTH = sys.argv[2]
else:
    dataFile = input("数据集地址：")

try:
    MAX_LENGTH = input("数据最大长度：")
    MAX_LENGTH = int(MAX_LENGTH)
except:
    if MAX_LENGTH > 0:
        pass
    else:
        MAX_LENGTH = 512

le = preprocessing.LabelEncoder()

if dataFile:
    df = pd.read_csv(dataFile)
    df.drop_duplicates()

print("数据集格式如下：")
print(df)

dataA = df["sent1"].squeeze().astype(str).values.tolist()
dataB = df["sent2"].squeeze().astype(str).values.tolist()
dataLabel = df["label"].squeeze().values.tolist()

# 获取label标签
le.fit(dataLabel)
print(le.classes_)

labels = list(le.classes_)
# 获取标签格式数据
tgt = le.transform(dataLabel)

print("labels", labels)
print("labels len：", len(labels))

# print(tgt)

inputsA = tokenizer(dataA, dataB, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
tgt = torch.Tensor(tgt)

# print(inputsA)
traindataset = TensorDataset(inputsA['input_ids'], inputsA['token_type_ids'], inputsA['attention_mask'],
                             tgt
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
