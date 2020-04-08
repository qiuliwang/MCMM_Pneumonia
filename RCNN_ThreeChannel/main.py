'''
Created by Wang Qiuli, Li Zhihuan
2019/4/8

wangqiuli@cqu.edu.cn
'''

from dataprepare import Data
from config import Config
import tensorflow as tf
import os
import numpy as np 
from tqdm import tqdm
from newmodel import RCNNMODEL
import csvTools
import random
import tensorflow.contrib.slim as slim
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
data = Data()

checkpoint_file = 'inception_v3_2016_08_28/inception_v3.ckpt'  
traintxt = csvTools.readCSV('./labels/train.csv')
testtxt = csvTools.readCSV('./labels/test.csv')
validtxt = csvTools.readCSV('./labels/valid.csv')
traindata = []

for one in traintxt:
    if len(one) != 0:
        
        traindata.append(one)

testdata=[]

for one in testtxt:
    if len(one) != 0:
        
        testdata.append(one)

validdata = []
for one in validtxt:
    if len(one) != 0:
        validdata.append(one)
config = Config()

load = True
load_cnn = False

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = RCNNMODEL(config)

    sess.run(tf.global_variables_initializer())   
    if load:
        model.load(sess, './models/19890.npy')
    if load_cnn:
        model.load_cnn(sess, './resnet50_no_fc.npy')

        # tf.global_variables_initializer().run()                                              
        # exclusions = ['Mixed_7c','Mixed_7b','AuxLogits','AuxLogits','Logits','Predictions','global_step']
        # variables_to_restore = slim.get_variables_to_restore(exclude=exclusions)
        # init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore, ignore_missing_vars=True)
        # init_fn(sess)        

    tf.get_default_graph().finalize()
    
    model.train(sess, traindata, validdata, data, False)

    model.train(sess, traindata, validdata, data, False)

    model.train(sess, traindata, validdata, data, False)
    
    model.test(sess, validdata, data, 'test')
    
