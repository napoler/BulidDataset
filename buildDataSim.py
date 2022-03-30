# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
自动构建数据集 预处理使用
seq2seq模式数据集
数据参考示例
dataDemo/Seq2seq.csv

"""
import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split

from config import *

# 输出目录
path = "out"
MAX_LENGTH = 128
print("""
seq2seq模式数据集
数据参考示例
dataDemo/Seq2seq.csv

""")
dataFile = input("数据集地址：")
if dataFile:
    df = pd.read_csv(dataFile)
    df.drop_duplicates()

dataA = df.iloc[:, [0]].squeeze().astype(str).values.tolist()
dataB = df.iloc[:, [1]].squeeze().astype(str).values.tolist()
# labels=df.iloc[:,[2]].squeeze().values.tolist()


inputsA = tokenizer(dataA, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
inputsB = tokenizer(dataB, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
# inputsLabels=torch.Tensor(labels)
traindataset = TensorDataset(inputsA['input_ids'], inputsA['attention_mask'],
                             inputsB['input_ids'], inputsB['attention_mask'],
                             # inputsLabels
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

torch.save(train, path + "/train.pkt")
torch.save(val, path + "/val.pkt")
torch.save(test, path + "/test.pkt")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')

    pass
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
