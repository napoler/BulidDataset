# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
自动构建数据集 预处理使用
基本的LM数据集模式数据集
数据参考示例
dataDemo/LM.csv

数据集没有做mask，
使用from tkitAutoMask import autoMask包来实现动态mask

"""
import os
import sys

import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split

from config import *

# 输出目录
path = "out"
MAX_LENGTH = 64

print("""
seq2seq模式数据集
数据参考示例
dataDemo/LM.csv

""")
if len(sys.argv) > 1:
    dataFile = sys.argv[1]
else:
    dataFile = input("数据集地址：")

if dataFile:
    df = pd.read_csv(dataFile)
    # 去重复
    # df.drop_duplicates()
print("数据集格式如下：")
print(df)
dataA = df.iloc[:, [0]].squeeze().astype(str).values.tolist()

inputsA = tokenizer(dataA, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
# print(inputsA)
# train_dataset = TensorDataset(inputsA['input_ids'], inputsA['token_type_ids'], inputsA['attention_mask'])
train_dataset = TensorDataset(inputsA['input_ids'], inputsA['attention_mask'])
fullLen = len(train_dataset)
trainLen = int(fullLen * 0.7)
valLen = int(fullLen * 0.15)
testLen = fullLen - trainLen - valLen
# print(fullLen)
train, val, test = random_split(train_dataset, [trainLen, valLen, testLen])

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
