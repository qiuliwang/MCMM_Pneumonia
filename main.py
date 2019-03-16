from dataprepare import Data
from config import Config
import tensorflow as tf
import os
import numpy as np 
from tqdm import tqdm
from newmodel import RCNNMODEL
import csvTools

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)

datapath = '/home/wangqiuli/Data/liver_cancer_dataset/train_dataset/'

labelpath = './train.csv'
data = Data(datapath, labelpath)
patients = data.patients
labels = data.labels

config = Config()

load = False
load_cnn = True


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = RCNNMODEL(config)

    sess.run(tf.global_variables_initializer())   
    if load:
        model.load(sess, './models/52801.npy')
    if load_cnn:
        model.load_cnn(sess, './resnet50_no_fc.npy')
    tf.get_default_graph().finalize()
    model.train(sess, patients, labels, data, False)
    testlabel = './test.csv'
    testlabel = csvTools.readCSV(testlabel)
    model.test(sess, testlabel, data)
    print('resnet50 with weight learning rate 0.01')