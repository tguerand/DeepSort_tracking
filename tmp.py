# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:00:34 2021

@author: trist
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def _xywh_to_xyxy(bbox_xywh, width, height):
        x, y, w, h = bbox_xywh
        x1 = x#max(int(x - w / 2), 0)
        x2 = x + w#min(int(x + w / 2), width - 1)
        y1 = y#max(int(y - h / 2), 0)
        y2 = y + h#min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2

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
        # x1 = int(x1 * coeffx1 + b)
        # y1 = int(y1 * coeffy1 + b)
        # x2 = int(x2 * coeffx2 + b)
        # y2 = int(y2 * coeffy2 + b)
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



p = r'C:\Users\trist\Documents\CS\3A\DL\project\DeepSort_tracking\output'
output_txt = os.path.join(p, 'output.txt')
p3 = r'C:\Users\trist\Documents\CS\3A\DL\project\DeepSort_tracking\det'
p2 = r'C:\Users\trist\Documents\CS\3A\DL\project\DeepSort_tracking\data\MOT16\train\MOT16-02\img1'

for i in tqdm(range(600)):
    if i < 9:
        mot_path = os.path.join(p2, '00000'+str(i+1)+r'.jpg')
    elif i < 99:
        mot_path = os.path.join(p2, '0000'+str(i+1)+r'.jpg')
    else:
        mot_path = os.path.join(p2, '000'+str(i+1)+r'.jpg')
    detections_path = os.path.join(p3, 'detections'+str(i)+'.txt') 
    mot_img = cv2.imread(mot_path)
    mot_img = cv2.resize(mot_img, (416, 416))
    w, h, _ = mot_img.shape
    dets = np.loadtxt(detections_path)
    
    detections = [_xywh_to_xyxy(bbox_xywh, w, h) for bbox_xywh in dets]
    new_img = draw_boxes(mot_img, detections)

    cv2.imwrite(os.path.join(p, 'originals', 'orig'+str(i)+'.jpg'), new_img)

# frame3 = os.path.join(p3, 'detections3.txt')
# outputs = np.loadtxt(frame3)
# #frame_idx, bboxes = outputs[:,0], outputs[:,2:6]


# img_path = os.path.join(p, 'frame3.jpg')
# #bboxes3 = bboxes[np.where(frame_idx == 3), :][0]


# img_path2 = os.path.join(p2, '000003.jpg')
# img = cv2.imread(img_path2)
# img = cv2.resize(img, (416, 416))
# w, h, _ = img.shape
# coeffs = np.zeros((13, 2))
# #coeffs[:,0] = bboxes3[:,0]/bboxes3[:,2]
# #coeffs[:,1] = bboxes3[:,1]/bboxes3[:,3]

# coeffx1, coeffy1, coeffx2, coeffy2 = 1, 1, 1, 1
# b = 0



