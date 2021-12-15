# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
自动构建数据集 预处理使用
Label Studio解析为ner数据集

"""
import json
import os
import random

import pandas as pd
import torch
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn import preprocessing
from torch.utils.data import random_split, TensorDataset
from pprint import pprint

from config import *
from libs import AutoClear
import tkitJson
import json
from tqdm.auto import tqdm
import numpy as np
# 输出目录
path = "out"
MAX_LENGTH = 512
FAKE_NUM=1
print("""
seq2seq模式数据集
数据参考示例
dataDemo/SequenceClassification.csv

""")
# dataFile = input("数据集地址：")
dataFile="data/data.json"
# if dataFile:
#     df = pd.read_csv(dataFile)
#     df.drop_duplicates()
# print("数据集格式如下：")
# print(df)
if dataFile:
    # Tjson=tkitJson.Json(dataFile)
    with open(dataFile,'r') as f:
        data=json.load(f)
else:
    exit()

# 构建标签
labels={"O":0}
i=0
bad=0


for it in tqdm(data):
    # print(it['result']['data']['value'])
    # print(it['result']['annotations'])
    # text = it['result']['data']['value']
    print(it['result'].keys())
    if ("annotations" in it['result'].keys()) and ("data" in it['result'].keys()):
        # for key in it['result'].keys():
        #     print(key)
        i+=1
        for iit in it['result']["annotations"]:
            # print(iit['result'])
            for iiit in iit['result']:
                # print(iiit)
                if iiit['type'] == "labels":
                    # print(iiit['value']['labels'])
                    for label in iiit['value']['labels']:
                        try:
                            labels[label]+=1
                        except:
                            labels[label]=0

    elif "nodeArr" in it['result'].keys():
        bad+=1
        pass
        # for key in it['result'].keys():
        #     print(key)

        #
        #
        #
        #
        #
        #
        #
        #
        #
        # print(it.keys())
        # print(it)
        #
        # print("==="*20)
        # print(it['result']["text"])
        # print(it['result']["nodeArr"])
        # # print(it['result']["nodeArr"])
        #
        #
        #
        # for iit in it['result']["nodeArr"]:
        #     print(iit)
        #     print(iit["text"])
        #     print(len(iit["text"]))
        #     for iiit in iit['result']:
        #         # print(iiit)
        #         if iiit['type'] == "labels":
        #             # print(iiit['value']['labels'])
        #             for label in iiit['value']['labels']:
        #                 try:
        #                     labels[label]+=1
        #                 except:
        #                     labels[label]=0
print("labels:")
pprint(labels)
print("Good",i)
print("Bad",bad)

    # break
labels_arr=list(labels.keys())
outdata={"text":[],"tags":[]}
CC=AutoClear()
for it in tqdm(data):
    # print(it['result']['data']['value'])
    # print(it['result']['annotations'])

    # print(it['result'].keys())
    if ("annotations" in it['result'].keys()) and ("data" in it['result'].keys()):
        # for key in it['result'].keys():
        #     print(key)
        # print(it['result']['data']['value'])
        # print(it['result'].keys())
        # FAKE_NUM
        for fi in range(FAKE_NUM):
            rand_num = random.randint(0, 128)
            text = it['result']['data']['value']
            text=CC.clearText(text)
            text="".join(rand_num*["ر"])+text
            # tags=["O"]*(len(text)+1)
            tags = [labels_arr.index("O")] *MAX_LENGTH
            # print(len(tags))


            for iit in it['result']["annotations"]:
                # print(iit['result'])
                for iiit in iit['result']:
                    # print(iiit)

                    if iiit['type'] == "labels":
                        # print(iiit)
                        # print(text[iiit['value']['start']:iiit['value']['end']])
                        try:
                            if text[(iiit['value']['start']+rand_num):(iiit['value']['end']+rand_num)]==iiit['value']['text'].lower():
                                # print(text[(iiit['value']['start']+rand_num):(iiit['value']['end']+rand_num)],iiit['value']['text'])
                                pass
                            else:
                                print("不一样")
                                print(text[(iiit['value']['start'] + rand_num):(iiit['value']['end'] + rand_num)],
                                      iiit['value']['text'].lower())
                                pass
                            # print(iiit['value']['start']+rand_num, iiit['value']['end']+rand_num)
                            # print(iiit['value']['start'], iiit['value']['end'])
                            for pos in range(iiit['value']['start']+rand_num, iiit['value']['end']+rand_num):
                                tags[pos]="I-"+iiit['value']['labels'][0]
                            tags[iiit['value']['start']+rand_num] = "B-" + iiit['value']['labels'][0]
                            tags[iiit['value']['end']-1+rand_num] = "E-" + iiit['value']['labels'][0]
                        except:
                            pass
            text=CC.clearTextDec(list(text))
            outdata["text"].append(" ".join(text))
            outdata["tags"].append(tags)
        # for w,t in zip(text,tags):
        #     print(w,t)
                    # print(iiit['value']['labels'])
                    # for label in iiit['value']['labels']:
                    #     try:
                    #         labels[label]+=1
                    #     except:
                    #         labels[label]=0

    elif "nodeArr" in it['result'].keys():
        pass


# print(outdata)

# MAX_LENGTH
tags =[]
for t in outdata["tags"]:
    tags.extend(t)
#


# # 构建标签
le = preprocessing.LabelEncoder()
le.fit(tags)

print(le.classes_)
labels = list(le.classes_)

try:
    os.makedirs(path)
except:
    pass
with open(path + "/labels.json", 'w', encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False)


# print(outdata["tags"][:1])
for i,it in enumerate(outdata["tags"]):
    tgt = le.transform(it)
    # print(tgt.tolist())
    # break
    outdata["tags"][i]=tgt.tolist()

# labels = list(le.classes_)
# print("labels", labels)
# print("labels len：", len(labels))
# # 获取标签格式数据
# tgt = le.transform(outdata["tags"])



# print(outdata["text"][:2])
# print(outdata["tags"][:1])
tokenizer = BertTokenizerFast.from_pretrained("tokenizer",
                                              return_offsets_mapping=True, model_max_length=1000000,
                                              do_basic_tokenize=False
                                              #   tokenize_chinese_chars=True

                                              )
inputsA = tokenizer(outdata["text"], return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
tgt = torch.Tensor(outdata["tags"])

traindataset = TensorDataset(inputsA['input_ids'],inputsA['token_type_ids'], inputsA['attention_mask'],
                             tgt
                             )

fullLen = len(traindataset)
trainLen = int(fullLen * 0.7)
valLen = int(fullLen * 0.15)
testLen = fullLen - trainLen - valLen

train, val, test = random_split(traindataset, [trainLen, valLen, testLen])



torch.save(train, path + "/train.pkt")
torch.save(val, path + "/val.pkt")
torch.save(test, path + "/test.pkt")