# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
peizhi
"""
from transformers import BertTokenizerFast

# tokenizer = BertTokenizerFast.from_pretrained("uer/chinese_roberta_L-2_H-128")
# tokenizer.save_pretrained("tokenizer")
tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
if __name__ == '__main__':
    pass
