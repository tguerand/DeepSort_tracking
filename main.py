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

import deepsort
from deepsort.model_encoder.encoder import Encoder
from deepsort.metrics import NearestNeighbor
from deepsort.kalman import KalmanFilter
#from deepsort.tracker import Tracker
#import deepsort

#Name of video file
videofile = r'.\data\MOT16\video.avi'

#Parameters of YOLO detector
confidence = float(0.4)
nms_thesh = float(0.4)
start = 0
reso = 416 #Image resolution
CUDA = torch.cuda.is_available()


def get_detections():
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
    file = r'./det/detections.txt'
    for row in d:
        np.savetxt(file, row)
    file.close()
    return d
# d is a list of all boxes detected in different video frames d[i] : coordinates of boxes detected in frame i 
#d : list of boxes
#d[i] : boxes found in frame i of the video 
#shape of d[i] = D x 4 
#D: number of persons in the frame
#4 =  (x_top_left, y_top_left, width, height)

def main(dfile_name=r'./det/detections.txt'):
    
    # Initialize
    metric = NearestNeighbor("cosine", 0.7)
    kf = KalmanFilter()
    if dfile_name is not None:
        detections = np.loadtxt(dfile_name)
    else:
        detections = get_detections()
    tracker = deepsort.tracker.Tracker(metric, kf)
    encoder = Encoder(r"./deepsort/checkpoints/ckpt.t7")
    video_path = video_file
    output_path = r'output.txt'
    
    out_path = r"./output"
    if not os.exists(out_path):
        os.mkdir(out_path)
        
    # Load video
    vid = cv2.VideoCapture(video_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    results = []
    i = 0
    while True:
        ret, frame = vid.read()
        dets = []
        print(i)
        if ret:
            
            bboxes = detections[i]
            dets.append([])
            for bbox in bboxes:
                x, y, w, h = bbox
                x1, y1 = max(int(x), 0), max(int(y), 0)
                x2, y2 = min(int(x + w), width -1), min(int(y + h), height -1)
                img_crop = frame[x1:y1,x2:y2]
                features = encoder(img_crop)
                dets.append(bbox, features)
            
            tracker.predict()
            tracker.update(dets)
            
            for track in tracker.tracks:
                if not track.state == 0 or track.time_since_update > 1:
                    continue
                bbox = track.get_position() # tlwh format
                results.append([frame_idx, track.track_id,
                                bbox[0], bbox[1], bbox[2], bbox[3]])
                
        else:
            break
        
        i+=1
        
    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
            
        
    
if __name__ == "__main__":
    #main(dfile_name=None)
    main()
