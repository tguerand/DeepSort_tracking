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
 
dir_path = r'C:\Users\trist\Documents\CS\3A\DL\project\DeepSort_tracking\output\frames'
out_path = r'C:\Users\trist\Documents\CS\3A\DL\project\DeepSort_tracking\output\output.avi'


    
img_array = []
for filename in glob.glob(os.path.join(dir_path , r'*.jpg')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 

out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in tqdm(range(len(img_array))):
    out.write(img_array[i])
out.release()