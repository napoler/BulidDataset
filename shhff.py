# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
假设我有一个包含内容的张量：
[0,55,49,38,2,0,96,28,2,0,73,2]
0 是句子标记的开头
2 是
我想要的句子标记的结尾做随机句子排列，如
[0,73,2,0,55,49,38,2,0,96,28,2]
或
[0,73,2,0,96,28,2,0,55 ,49,38,2]
我该怎么做？
"""

import torch

"""

"""
# https://discuss.pytorch.org/t/how-to-shuffle-sentence-based-on-ids/105336
x = torch.tensor([0, 55, 49, 38, 2, 0, 96, 28, 2, 0, 73, 2])
idx = torch.cat((torch.tensor([0]), torch.randperm(len(x) - 2) + 1, torch.tensor([len(x) - 1])))
print(idx)
out = x[idx]
print(out)

if __name__ == '__main__':
    pass
