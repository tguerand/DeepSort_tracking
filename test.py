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

from deepsort.model_encoder.encoder import Encoder
#from deepsort.metrics import NearestNeighbor
#from deepsort.kalman import KalmanFilter
#from deepsort.tracker import Tracker
import deep_sort
#import deepsort

#Name of video file
videofile = r'.\data\MOT16\video.avi'

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
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def _xywh_to_xyxy(bbox_xywh, width, height):
        x, y, w, h = bbox_xywh
        x1 = int(x)#max(int(x - w / 2), 0)
        x2 = int(x + w)#min(int(x + w / 2), width - 1)
        y1 = int(y)#max(int(y - h / 2), 0)
        y2 = int(y + h)#min(int(y + h / 2), height - 1)
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
    
    
    output_path = r'./output/output.txt'
    save_txt = True
    save_vid = True
    save_path = r'./output/output.avi'
    vid_writer = None
    
    out_path = r"./output"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    # Load video
    vid = cv2.VideoCapture(video_path)
    
    deep = deep_sort.DeepSort(r"./deepsort/checkpoints/new_ckpt.t7")
    
    results = []
    i = 0
    while True:
        ret, frame = vid.read()
        dets = []
        print(i)
        
        
        
        if ret:
            frame = cv2.resize(frame, (416, 416))
            
            
            bboxes = detections[i]
            
            
            if len(bboxes) == 0:
                deep.increment_ages()
            
            width, height = frame.shape[:2]
            features, box_left_idx = _get_features(bboxes, frame, width, height, encoder)
            
            for idx in box_left_idx:
                bboxes = np.delete(bboxes, idx, axis=0)

            dets = [[bboxes[j], features[j]] for j in range(len(bboxes))]
            
            outputs = deep.update(bboxes, scores, frame)
            
            # draw boxes for visualization
            bbox_xyxy = []
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                #bbox_xyxy = [_xywh_to_xyxy(bbox[:4], width, height) for bbox in outputs]
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
        
    # Store results.
    # f = open(output_path, 'w')
    # for row in results:
    #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
    #         row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    # f.close
    img2vid.img2video(out_path, r'./output/output.avi')
        
    
if __name__ == "__main__":
    #main(dfile_name=None)
    main(videofile)#, dfile_name=None)
