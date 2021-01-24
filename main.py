# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:15:49 2021

@author: trist
"""

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
import img2vid
import json
import reconstruct

from deepsort.model_encoder.encoder import Encoder
#from deepsort.metrics import NearestNeighbor
#from deepsort.kalman import KalmanFilter
#from deepsort.tracker import Tracker
import deep_sort
#import deepsort

#Name of video file
videofile = r'.\data\MOT16\video.avi'

#Parameters of YOLO detector
confidence = float(0.5)
nms_thesh = float(0.4)
start = 0
reso = 416 #Image resolution
CUDA = torch.cuda.is_available()

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def _tlwh_to_xyxy(bbox_xywh, width, height):
        x, y, w, h = bbox_xywh
        x1 = int(x)#max(int(x - w / 2), 0)
        x2 = int(x + w)#min(int(x + w / 2), width - 1)
        y1 = int(y)#max(int(y - h / 2), 0)
        y2 = int(y + h)#min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2

def _tlwh_to_xywh(bbox_tlwh):
    x, y, w, h = bbox_tlwh
    x_c = int(x + w/2)
    y_c = int(y + h/2)
    return [x_c, y_c, w, h] 


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

def get_config(config_path):
    dico = {}
    with open(config_path) as file:
        dico = json.load(file)
    return dico


def main(video_path, dfile_name=r'./det/dets/', config_path='./cfg/config.json'):
    
    # Initialize
    if dfile_name is not None:
        files = []
        detections = []
        for file in os.listdir(dfile_name):
            if file.endswith('.txt'):
                files.append(file)
                
                detections.append(np.array(np.loadtxt(os.path.join(dfile_name, file))))
    else:
        detections = get_detections()
    
    args = get_config(config_path)
    output_path = r'./output/output.txt'
    save_txt = True
    save_vid = True
    save_path = r'./output/output.avi'
    
    
    out_path = r"./output"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    # Load video
    vid = cv2.VideoCapture(video_path)
    
    deep = deep_sort.DeepSort(r"./deepsort/checkpoints/new_ckpt.t7",
                              max_dist=args['MAX_DIST'],
                              min_confidence=args['MIN_CONFIDENCE'],
                              nms_max_overlap=args['NMS_MAX_OVERLAP'],
                              max_iou_distance=args['MAX_IOU_DISTANCE'],
                              max_age=args['MAX_AGE'],
                              n_init=args['N_INIT'],
                              nn_budget=args['NN_BUDGET'],
                              use_cuda=True)
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    
    i = 0
    while True:
        ret, frame = vid.read()

        print(i)
        
        if ret:
            frame = cv2.resize(frame, (416, 416))
            
            bboxes, scores = detections[i][:,:4], detections[i][:,4]
            bboxes = [_tlwh_to_xywh(bbox) for bbox in bboxes]
            
            if len(bboxes) == 0:
                deep.increment_ages()
            
            width, height = frame.shape[:2]
            
            
            for idx, box in enumerate(bboxes):
                if box == [0, 0, 0, 0]:
                    bboxes = np.delete(bboxes, idx, axis=0)
                    scores = np.delete(scores, idx)

            
            outputs = deep.update(bboxes, scores, frame)
            
            # draw boxes for visualization
            bbox_xyxy = []
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                
                identities = outputs[:, -1]
                draw_boxes(frame, bbox_xyxy, identities)

            # Write MOT compliant results to file
            if save_txt and len(outputs) != 0:
                for j, output in enumerate(outputs):
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2]
                    bbox_h = output[3]
                    identity = output[-1]
                    with open(output_path, 'a') as f:
                        f.write(('%g ' * 10 + '\n') % (i, identity, bbox_left,
                                                       bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
        
            if save_vid:
                save_path = out_path + r"/frames/frame" + str(i) + ".jpg"
                cv2.imwrite(save_path, frame)

        
        
        else:
            break
        
        i+=1
        
    # create video
    reconstruct.reconstruct(r'./data/MOT16/train/MOT16-02/img1',
                            output_path,
                            out_path=r'./output/output_iou_09.avi')
        
    
if __name__ == "__main__":
    
    main(videofile)#, dfile_name=None)
