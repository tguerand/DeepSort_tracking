# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 12:09:53 2021

@author: trist
"""

import cv2
import os
import numpy as np
import glob
from tqdm import tqdm

dir_path = r'./data/MOT16/train/MOT16-02/img1'
results_path = r'./output/output.txt'

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

def reconstruct(dir_path, results_path, out_path='./output/recon.avi'):
    
    results = np.loadtxt(results_path)
    frames = results[:,0]
    bboxes = results[:, 2:6]
    identities = results[:, 1]
    
    idx_advance = 60
    
    img_array = []
    for filename in glob.glob(os.path.join(dir_path , r'*.jpg')):
        img = cv2.imread(filename)
        img = cv2.resize(img, (416, 416))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
    for i, frame in enumerate(img_array):
        idx = np.where(frames == i+1)
        bbox_xyxy = bboxes[idx]
        identity = identities[idx]
        real_idx = (i + idx_advance)%len(img_array)
        draw_boxes(img_array[real_idx], bbox_xyxy, identity)
     
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release()