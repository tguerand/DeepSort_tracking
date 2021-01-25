# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:55:46 2021

@author: trist
"""

	
import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
 
dir_path = r'.\data\caltech_extracted\set01\V000\images'
out_path = r'.\data\set01_000.avi'


def im2vid(dir_path, out_path):
    
    img_array = []
    for filename in glob.glob(os.path.join(dir_path , r'*.jpg')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        the_size = (width,height)
        img_array.append(img)
     
     
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, the_size)
     
    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release()