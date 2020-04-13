# -*- coding:utf-8 -*-

import os
import csvTools
import pydicom
import numpy as np
import math
import scipy.misc
import operator
cmpfun = operator.attrgetter('InstanceNumber')
from PIL import Image
from random import choice
import cv2
import rotate
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array, max, min):
    image = image_array.copy()
    image[image > max] = max
    image[image < min] = min
    image = normalazation(image)
    return image

def getThreeChannel(pixhu):
    lungwindow = truncate_hu(pixhu, 400, -1000)
    highattenuation = truncate_hu(pixhu, 240, -160)
    lowattenuation = truncate_hu(pixhu, -950, -1400)
    pngfile = [lowattenuation, lungwindow, highattenuation]
    pngfile = np.array(pngfile).transpose(1,2,0)
    return  pngfile  

# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def get_pixels_hu(ds):
    image = ds.pixel_array
    image = np.array(image , dtype = np.float32)
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    image = image * slope
    image += intercept
    return image
    
def gray2rgb(im):
    im=im[:,:,np.newaxis]
    im0=im1=im
    
    im=np.concatenate((im0,im1,im),axis=2)
    return im

def angle_transpose(file,degree):
    '''
     @param file : a npy file which store all information of one cubic
     @param degree: how many degree will the image be transposed,90,180,270 are OK
    '''
    array = file

    newarr = np.zeros(array.shape,dtype=np.float32)
    for depth in range(array.shape[0]):
        jpg = array[depth]
        jpg.reshape((jpg.shape[0],jpg.shape[1],1))
        img = Image.fromarray(jpg)
        out = img.rotate(degree)
        newarr[depth,:,:] = np.array(out).reshape(array.shape[1],-1)[:,:]
    return newarr

class Data(object):
    def __init__(self):
        self.wordModel = Word2Vec.load('./txt/word2vec.model')
        self.transformedlistd = csvTools.readCSV('./txt/transformedlistd.csv')
        sentences = []

        for onecontext in self.transformedlistd:
            words = onecontext[3].split(' ')
            sentences.append(words)

        model = Word2Vec(sentences, size=50, window=3, min_count=1, workers=2)
        self.model = model
        vector = model['上腹', '疼痛', '周']

    def getOnePatient(self, patientName, transsign = False):
        model = self.wordModel
        transformedlistd = self.transformedlistd
        if 'fung' in patientName[1]:
            datapath = '/raid/data/pneumonia/FungusScreened/'
        elif 'bact' in patientName[1]:
            datapath = '/raid/data/pneumonia/BacteriaScreened/'
        else:
        
            datapath = '/raid/data/pneumonia/NormalScreened/'
        dcmfiles = os.listdir(datapath + patientName[0])
        dcmfiles.sort()
        slices = [pydicom.dcmread(os.path.join(datapath, patientName[0], s)) for s in dcmfiles]
        slices.sort(key = cmpfun)
        slicethickness = slices[0].data_element('SliceThickness').value
        dcmkeep = []
        keeprate = 10 / slicethickness
        keeprate = int(math.floor(keeprate))
        if keeprate < 1:
            keeprate = 1

        tempsign = 0
        for onedcm in slices:
            if tempsign % keeprate == 0:
                dcmkeep.append(onedcm)
            tempsign += 1
        if len(dcmkeep) > 32:
            
            dcmkeep = dcmkeep[:32]
        if len(dcmkeep) < 32:
            temp = []
            for i in range(0, 32 - len(dcmkeep)):
                temp.append(dcmkeep[0])
            dcmkeep = temp + dcmkeep

        
        indexlist = [0,1,2,3]

        if transsign == True:
            index = choice(indexlist)
        else:
            index = 0
        pixels = []
        mid = dcmkeep[len(dcmkeep) // 2]
        angle = rotate.rotate_angle(mid.pixel_array)
        if angle > 30 or angle < -30:
            angle = 0
        sign = 1
        for temp in dcmkeep:
            temp = get_pixels_hu(temp)
            temp=getThreeChannel(temp)
            temp = cv2.medianBlur(temp, 3)
            sign += 1
            pixels.append(temp)
            

        pixels = np.array(pixels, dtype=np.float)
 
        if index == 1:
            pixels = angle_transpose(pixels, 90)
        if index == 2:
            pixels = angle_transpose(pixels, 180)        
        if index == 3:
            pixels = angle_transpose(pixels, 270)
        if len(pixels.shape)<4:
            pixels = np.expand_dims(pixels, -1)        
   
        #get id sex and complaint
        id = patientName[0]
        id = id[:id.index('_')]
        
        patientcomplaint = []
        sex = 0
        age = 0
        for oneinfo in transformedlistd:
            if str(id) == str(oneinfo[0]):
                sex = oneinfo[1]
                age = oneinfo[2]
                age = float(age) / 100
                patientcomplaint = oneinfo[3].split(' ')
                break
        if len(patientcomplaint) < 16:
            for i in range(0, 16 - len(patientcomplaint)):
                patientcomplaint.append('无')

        wordsvec = self.model[patientcomplaint[:16]]
        return pixels, sex, age, wordsvec

    def getThickness(self):
        return self.slicethicknesscount

