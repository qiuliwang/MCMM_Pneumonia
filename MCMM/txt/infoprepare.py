# -*- coding:utf-8 -*-

from pyltp import SentenceSplitter
from pyltp import Segmentor
import csvTools
import re
from string import digits
from zhon.hanzi import punctuation
from tqdm import tqdm
import os


contexts = csvTools.readCSV('transformedlistd.csv')

complaint = ""
for onecontext in contexts:
    temp = onecontext[-1]
    # print(temp)
    temp += ' '
    complaint += temp
# print(complaint)

source_vocab = list(set(complaint.split()))
source_int_to_vocab = {idx: word for idx, word in enumerate(source_vocab)}

vocab = []
# print(source_int_to_vocab)
for one in source_int_to_vocab:
    key = one
    content = source_int_to_vocab[one]
    temp = [key, content]
    vocab.append(temp)
csvTools.writeCSV('vocab.csv', vocab)