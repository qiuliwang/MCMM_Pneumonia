import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import SimpleITK as sitk
from skimage import transform
import os
import matplotlib.pyplot as plt
def rotate_angle(mid):
    w,h = mid.shape
    im = mid
    ret,img=cv2.threshold(im, -100, 1, cv2.THRESH_BINARY)
    _,contours, hierarchy = cv2.findContours(img.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_shape=[i.shape[0] for i in contours]
    max_contour_index=np.argmax(contours_shape)
    min_Rect=cv2.minAreaRect(contours[max_contour_index])
    angle=min_Rect[2]
  
    return angle
def get_max_rotate_angle(dcm_series):
    w,h,c=dcm_series.shape
    lis=[]
    for i in range(c):
        im=dcm_series[:,:,i]
        ret,img=cv2.threshold(im, -100, 1, cv2.THRESH_BINARY)
        _,contours, hierarchy = cv2.findContours(img.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_shape=[i.shape[0] for i in contours]
        max_contour_index=np.argmax(contours_shape)
        min_Rect=cv2.minAreaRect(contours[max_contour_index])
        angle=min_Rect[2]
        lis.append(angle)
    lis1=np.abs(lis)
    ind=np.argmax(lis1)
  
    return lis[ind]
def rotate(dcm_slice,rotate_angle):
        w,h=dcm_slice.shape

        M=cv2.getRotationMatrix2D((w/2,h/2),rotate_angle,1)
        rotate=cv2.warpAffine(dcm_slice,M,(w,h))
        
        return rotate
        
        
    




