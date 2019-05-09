# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:45:48 2018

@author: lizhihuan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:16:10 2018

@author: lizhihuan
"""
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from histogram import  get_seed 
from floodfill import close_
import os
import time
#file_path='/raid/data/clinic/data/nii_file/'
#img=nib.load(r"C:\Users\lizhihuan\Desktop\slic\test.nii") 
#img_arr=img.get_fdata()
#img=np.squeeze(img_arr)
#img=np.load(r'/home/lizhihuan/slic/blur_img.npy')

class Point(object):
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z=z
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getZ(self):
        return self.z
    
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y,currentPoint.z]) - int(img[tmpPoint.x,tmpPoint.y,tmpPoint.z]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1,0), Point(0, -1,0), Point(1, -1,0), Point(1, 0,0), Point(1, 1,0), \
                    Point(0, 1,0), Point(-1, 1,0), Point(-1, 0,0),Point(-1,-1,1),Point(0,-1,1),Point(1,-1,1),\
                    Point(1,0,1),Point(1,1,1),Point(0,1,1),Point(-1,1,1),Point(-1,0,1),Point(0,0,1),Point(-1,-1,-1), \
                    Point(0,-1,-1),Point(1,-1,-1),Point(1,0,-1),Point(1,1,-1),Point(0,1,-1),Point(-1,1,-1),Point(-1,0,-1), \
                    Point(0,0,-1)]
    else:
        connects = [   Point(1, 0,0),Point(0, 1,0), Point(-1, 0,0),\
                      Point(1, 0,-1),Point(0, 1,-1), Point(-1, 0,-1),\
                    Point(0, -1,1),  Point(1, 0,1),Point(0, 1,1), Point(-1, 0,1)]
    return connects
 
def regionGrow(img,grad,seeds,thresh,p =0):#30
    height, weight,slices = img.shape
    seedMark = np.zeros(img.shape,dtype=np.int)
    seedList = []
    label=1
    n=1
    sum=0
    connects = selectConnects(p)
    mean=-800
    for seed in seeds:
        
        seedList.append(seed)
        
        for i in range(len(connects)):
            tmpX = seed.x + connects[i].x
            tmpY = seed.y + connects[i].y
            tmpZ=seed.z + connects[i].z
            if tmpX < 0 or tmpY < 0 or tmpZ<0 or tmpX >= height or tmpY >= weight or tmpZ>=slices :#or grad[tmpX,tmpY,tmpZ]>200000:
                continue
            if (img[tmpX,tmpY,tmpZ]>=-850)&(img[tmpX,tmpY,tmpZ]<=-700):
                seedMark[tmpX,tmpY,tmpZ] = label
                seedList.append(Point(tmpX,tmpY,tmpZ))
    
    
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y,currentPoint.z] = label
        for i in range(len(connects)):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            tmpZ=currentPoint.z + connects[i].z
            if tmpX < 0 or tmpY < 0 or tmpZ<0 or tmpX >= height or tmpY >= weight or tmpZ>=slices :#or grad[tmpX,tmpY,tmpZ]>200000:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY,tmpZ))
            if seedMark[tmpX,tmpY,tmpZ] == 0 and grayDiff < thresh  and abs(img[tmpX,tmpY,tmpZ]-mean)<300:
                seedMark[tmpX,tmpY,tmpZ] = label
                sum=sum+img[tmpX,tmpY,tmpZ]
                mean=sum/n
                n=n+1
                seedList.append(Point(tmpX,tmpY,tmpZ))
    return seedMark
    
def lung_seg(lung_3d,thresh=30):
    img_arr=lung_3d.astype(np.float32)
    blurs=np.zeros_like(img_arr)
    for ii in range(img_arr.shape[2]):
        blurs[:,:,ii] = cv2.bilateralFilter(img_arr[:,:,ii],20,200,10)
    
    seed0,seed1=get_seed(img_arr,'name',min_HU=-950,max_HU=-750)
    print(seed0,seed1)
    seeds = [Point(seed0[0],seed0[1],seed0[2]),Point(seed1[0],seed1[1],seed1[2])]
    binaryImg = regionGrow(blurs,None,seeds,thresh).astype(np.int8)
    mask=close_(binaryImg,multi_floodfill=False)
    return mask
    

# if __name__ == '__main__':
#     file_path='/raid/data/lizhihuan/VESSEL12/'
#     for i in os.listdir(file_path):
#         i='abcde'
#         img_path=os.path.join(file_path,i)

#         img_arr=np.load('/home/lizhihuan/a.npy').astype(np.float32)
   
#         name_=i[:-4]
#         print(name_)
       
#         mask=lung_seg(img_arr,20)

#         np.save('/raid/data/lizhihuan'+'/mask_file/'+name_+'1.npy',mask)

