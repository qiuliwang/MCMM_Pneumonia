from dataprepare import Data
from config import Config
import tensorflow as tf
import os
import numpy as np 
from tqdm import tqdm
from newmodel import RCNNMODEL
import csvTools
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)



def div_list(ls,n):
    '''
    divide ls into n folders
    ls -> list
    n -> int
    '''
    if not isinstance(ls,list) or not isinstance(n,int):
        return []
    ls_len = len(ls)
    if n<=0 or 0==ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len//n
        k = ls_len%n
        ls_return = []
        for i in range(0,(n-1)*j,j):
            ls_return.append(ls[i:i+j])
        ls_return.append(ls[(n-1)*j:])
        return ls_return

data = Data()
alldata = csvTools.readCSV('/raid/data/wangqiuli/Documents/pneumonia/label/all.csv')
dividedall = div_list(alldata, 5)
# print(len(dividedall[0]))
# for onep in dividedall[0]:
#     print(onep)

# traindata = dividedall[0] + dividedall[1] +dividedall[2] +dividedall[3]
# testdata = dividedall[4]
# print(len(traindata))
# print(len(testdata))

import multiprocessing
import time
import thread


npypath = '/home/wangqiuli/raid/pneumonia/numpyfiles/'
npyfiles = os.listdir(npypath)
dirpath = './segmask/'
import seg

# for onenpy in tqdm(npyfiles):
#     # print onenpy
#     temp = np.load(npypath + onenpy)
#     shape = temp.shape
#     # print shape
#     if shape[0] != 512:
#         print onenpy

def getMask(data):
    for onepatient in tqdm(data):
        try:
            pix = np.load(npypath + onepatient)
            pix = seg.lung_seg(pix)    
            np.save(dirpath + onepatient, pix)
        except:
            print(onepatient)
# getMask(testnpy)

def savenpyfordicom(alldata):
    for onepatient in tqdm(alldata):
        data.getOnePatient(onepatient)

savenpyfordicom(alldata[:20])
# dividedall = div_list(npyfiles, 5)
# p1 = multiprocessing.Process(target = getMask, args = (dividedall[0],))
# p2 = multiprocessing.Process(target = getMask, args = (dividedall[1],))
# p3 = multiprocessing.Process(target = getMask, args = (dividedall[2],))
# p4 = multiprocessing.Process(target = getMask, args = (dividedall[3],))
# p5 = multiprocessing.Process(target = getMask, args = (dividedall[4],))

# p1.start()
# p2.start()
# p3.start()
# p4.start()
# p5.start()

print(len(npyfiles))
dividedall = div_list(npyfiles, 10)

plist = []

for i in range(10):
    p = multiprocessing.Process(target = getMask, args = (dividedall[i],))
    plist.append(p)

for i in range(10):
    plist[i].start()


    # print(pix.shape)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#     model = RCNNMODEL(config)

#     sess.run(tf.global_variables_initializer())   
#     if load:
#         model.load(sess, './models/15900.npy')
#     if load_cnn:
#         model.load_cnn(sess, './resnet50_no_fc.npy')
#     tf.get_default_graph().finalize()
#     # model.train(sess, patients, labels, data, False)
#     testlabel = './csvfiles/test.csv'
#     testlabel = csvTools.readCSV(testlabel)
#     model.test(sess, testlabel, data)
#     print('resnet50 with weight learning rate 0.01, rnn units 512, dropout 0.5, one loss, 3 rnn layers')