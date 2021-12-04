# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
自动构建数据集 预处理使用
SequenceClassification模式数据集
数据参考示例
dataDemo/SequenceClassification.csv

"""
import json

import pandas as pd
import torch
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn import preprocessing
from torch.utils.data import random_split, TensorDataset

from config import *

# 输出目录
path = "out"
MAX_LENGTH = 128

le = preprocessing.LabelEncoder()
print("""
seq2seq模式数据集
数据参考示例
dataDemo/SequenceClassification.csv

""")
dataFile = input("数据集地址：")
if dataFile:
    df = pd.read_csv(dataFile)
    df.drop_duplicates()
print("数据集格式如下：")
print(df)
dataA = df.iloc[:, [0]].squeeze().astype(str).values.tolist()
dataB = df.iloc[:, [1]].squeeze().values.tolist()

le.fit(dataB)
labels = list(le.classes_)
print("labels", labels)
print("labels len：", len(labels))
# 获取标签格式数据
tgt = le.transform(dataB)
# print(tgt)

inputsA = tokenizer(dataA, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
tgt = torch.Tensor(tgt)

traindataset = TensorDataset(inputsA['input_ids'], inputsA['attention_mask'],
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

with open(path + "/labels.json", 'w', encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False)

torch.save(train, path + "/train.pkt")
torch.save(val, path + "/val.pkt")
torch.save(test, path + "/test.pkt")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')

    pass
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
