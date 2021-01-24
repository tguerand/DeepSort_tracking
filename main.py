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


from deepsort.model_encoder.encoder import Encoder
from deepsort.metrics import NearestNeighbor
from deepsort.kalman import KalmanFilter
from deepsort.tracker import Tracker

#import deepsort

#Name of video file
videofile = r'.\data\MOT16\video.avi'

#Parameters of YOLO detector
confidence = float(0.5)
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
    d =[] #list of box coordinates
    s =[] # list of box scores
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret :
            # detector is a D x 4 : all boxes detected in the frame
            dett = detector_video( frame, inp_dim, model, confidence, num_classes, nms_thesh, CUDA)
            detector = np.array(dett[:,:4]) # coordinates of boxes
            scores = np.array(dett[:,4]) # scores of boxes
            d_s = np.zeros((detector.shape[0], 5))
            d_s[:,:4] = detector
            d_s[:,4] = scores
            
            d.append(detector)
            s.append(scores)
            file = open('./det/dets/detections'+str(i)+'.txt','w')
            np.savetxt(file, d_s, fmt="%1.3f")
            file.close()
        else : 
            break 
        i+=1
    
    return d, s
# d is a list of all boxes detected in different video frames d[i] : coordinates of boxes detected in frame i 
#d : list of boxes
#d[i] : boxes found in frame i of the video 
#shape of d[i] = D x 4 
#D: number of persons in the frame
#4 =  (x_top_left, y_top_left, width, height)

def _xywh_to_xyxy(bbox_xywh, width, height):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2


def _get_features(bbox_xywh, ori_img, width, height, encoder):
        im_crops = []
        box_left_idx = []
        for i, box in enumerate(bbox_xywh):
    
            if (box == [0, 0, 0, 0]).all():
                box_left_idx.append(i - len(bbox_xywh))
                continue
            x1, y1, x2, y2 = _xywh_to_xyxy(box, width, height)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
            
        
        if im_crops:
            features = encoder(im_crops)
        else:
            features = np.array([])
        return features, box_left_idx


def main( video_path, dfile_name=r'./det'):
    
    # Initialize
    metric = NearestNeighbor("cosine", 0.7)
    kf = KalmanFilter()
    if dfile_name is not None:
        files = []
        detections = []
        for file in os.listdir(dfile_name):
            if file.endswith('.txt'):
                files.append(file)
                
                detections.append(np.array(np.loadtxt(os.path.join(dfile_name, file))))
    else:
        detections = get_detections()
    my_tracker = Tracker(metric, kf)
    encoder = Encoder(r"./deepsort/checkpoints/ckpt.t7")
    
    output_path = r'./output/output.txt'
    
    out_path = r"./output"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    # Load video
    vid = cv2.VideoCapture(video_path)
    
    
    results = []
    i = 0
    while True:
        ret, frame = vid.read()
        dets = []
        print(i)
        
        frame = np.reshape(frame, (reso,reso, frame.shape[-1]))
        # frame /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if frame.ndimension() == 3:
        #     frame = img.unsqueeze(0)
        
        if ret:
            
            bboxes = detections[i]
            
            width, height = frame.shape[:2]
            features, box_left_idx = _get_features(bboxes, frame, width, height, encoder)
            
            for idx in box_left_idx:
                bboxes = np.delete(bboxes, idx, axis=0)

            dets = [[bboxes[j], features[j]] for j in range(len(bboxes))]
            
            my_tracker.predict()
            my_tracker.update(dets)
            
            for track in my_tracker.tracks_list:
                if not track.state == 0 or track.age_update > 1:
                    continue
                print('aa')
                bbox = track.get_position() # tlwh format
                results.append([i, track.track_id,
                                bbox[0], bbox[1], bbox[2], bbox[3]])
                
        else:
            break
        
        i+=1
        
    # Store results.
    f = open(output_path, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
            
        
    
if __name__ == "__main__":
    #main(dfile_name=None)
    main(videofile, dfile_name=None)
