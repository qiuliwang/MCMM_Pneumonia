# -*- coding:utf-8 -*-

from pyltp import SentenceSplitter
from pyltp import Segmentor
import csvTools
import re
from string import digits
from zhon.hanzi import punctuation
from tqdm import tqdm
import os

segmentor = Segmentor()
segmentor.load_with_lexicon('./ltp/cws.model','lexicon.txt')

contexts2018 = csvTools.readCSV('2018all.csv')
contexts2017 = csvTools.readCSV('2017.csv')
contexts2016 = csvTools.readCSV('2016.csv')

contexts = contexts2016+contexts2017+contexts2018
print(len(contexts))

# birthlist = []
descriptionlist = []
dignosislist = []
for context in contexts:
    # birth = context[3]
    description = context[6]
    dignosis = context[7]
    # birthlist.append(birth)
    descriptionlist.append(description)
    dignosislist.append(dignosis)

chinesefilelist = os.listdir('chinese')
chinestcontentlist = []
for one in chinesefilelist:
    temp = csvTools.readTXT('chinese/' + one)
    chinestcontentlist += temp

chinese = []
# for one in chinestcontentlist:
#     if len(one) >2:
#         # print(one)
#         chinese.append(one)
# print len(chinese)
# # print(len(contextsdict))

trainlist = chinese + descriptionlist + dignosislist
import random
random.shuffle(trainlist)
print('len of trainlist: ', len(trainlist))
# print(len(trainlist))
count1 = 0
count2 = 0
f = open('corpus.txt', 'wr')

for item in tqdm(trainlist):
    # print '*****************'
    item = item.translate(None, digits)
    string = item
    # string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，？、~@#￥%……&*（）]+\:\：/-\-".decode("utf8"), "",item)
    # string = filter(lambda ch: ch not in ' \t1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', string) 
    sents = SentenceSplitter.split(string)  # 分句

    for sen in sents:
        sen = re.sub(ur"[%s]+" %punctuation, "", sen.decode("utf-8"))
        sen = sen.encode("utf-8")
        # print sen
        words = segmentor.segment(sen)
        # print ' '.join(words)
        tempinfo = ' '.join(words) + '\n'
        if len(words) > 1:
            f.write(tempinfo)
        wordslen = len(words)
        if wordslen <= 35:
            count1 += 1
        else:
            count2 += 1

f.close()
