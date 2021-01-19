from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

#Name of video file
videofile = 'venice1.mp4'

#Parameters of YOLO detector
confidence = float(0.4)
nms_thesh = float(0.4)
start = 0
reso = 416 #Image resolution
CUDA = torch.cuda.is_available()



## Load data
cap = cv2.VideoCapture(videofile)  
assert cap.isOpened(), 'Cannot capture source'

num_classes = 80    #For COCO
classes = load_classes("data/coco.names.txt") 


#Set up the neural network
print("Loading network.....")
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
print("Network successfully loaded")

model.net_info["height"] = reso# Image resolution 
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

model.eval()

frames = 0  
start = time.time()
d =[]
while cap.isOpened():
    ret, frame = cap.read()
    if ret :
        # detector is a D x 4 : all boxes detected in the frame
        detector = detector_video( frame, inp_dim, model, confidence, num_classes, nms_thesh, CUDA)
        d.append(detector)
    else : 
        break 
# d is a list of all boxes detected in different video frames d[i] : coordinates of boxes detected in frame i 