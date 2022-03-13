



import torch

import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
#判断是否用分隔符
import unicodedata
import re
# import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import *



class DatasetPruning(Dataset):
                       #      序号    关系
    def __init__(self, data, rel2idx, idx2rel):
        self.data = data
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = 'roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights, cache_dir='.')

    def __len__(self):
        return len(self.data)
           #TODO 词句的填充
    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr
          #TODO 将答案转为one-hot
          #TODO 转换为one-hot ，用序号编写one-hot
    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.rel2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot
            #todo  将问题词表转为填充好的张量，已经添加特殊词
    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:   #句子部分为1 ，空余部分为0
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[0]#TODO 将问题通过词表转为填充的张量
        question_tokenized, attention_mask = self.tokenize_question(question_text)

        #TODO 这是关系的序号
        rel_ids = data_point[1]
        #TODO 将关系序号转为one-hot
        rel_onehot = self.toOneHot(rel_ids)
        return question_tokenized, attention_mask, rel_onehot


def _collate_fn(batch):
    question_tokenized = batch[0]
    attention_mask = batch[1]
    rel_onehot = batch[2]
    print(len(batch))
    question_tokenized = torch.stack(question_tokenized, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    return question_tokenized, attention_mask, rel_onehot

class DataLoaderPruning(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderPruning, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

