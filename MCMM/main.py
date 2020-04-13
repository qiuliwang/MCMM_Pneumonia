'''
Modified by Wang Qiuli, Li Zhihuan

2019/5/9

We use some code from project: https://github.com/DeepRNN/image_captioning

main.py is entry of rcnn
'''

from dataprepare import Data
from config import Config
import tensorflow as tf
import os
from newmodel import RCNNMODEL
import csvTools
import tensorflow.contrib.slim as slim
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
data = Data()

checkpoint_file = 'inception_v3_2016_08_28/inception_v3.ckpt'  
traintxt = csvTools.readCSV('./labels/cross1.csv') + csvTools.readCSV('./labels/cross2.csv') + csvTools.readCSV('./labels/cross3.csv') + csvTools.readCSV('./labels/cross4.csv')
testtxt = csvTools.readCSV('./labels/cross5.csv')
validtxt = csvTools.readCSV('./labels/cross5.csv')
traindata=[]
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

load = False
load_cnn = True

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = RCNNMODEL(config)
    sess.run(tf.global_variables_initializer())   
    if load:
        model.load(sess, './models/10800.npy')
    if load_cnn:
        model.load_cnn(sess, './resnet50_no_fc.npy')
    

    tf.get_default_graph().finalize()

    model.train(sess, traindata, validdata, data, False)

    model.train(sess, traindata, validdata, data, False)

    model.train(sess, traindata, validdata, data, False)

    #model.train(sess, traindata, validdata, data, False)
    
    model.test(sess, validdata, data, 'test')
