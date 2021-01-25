# DeepSort_tracking
Centralesupélec Deep Learning Course project on multi object tracking with DeepSort

## Yolo
To detect pedestrians in a video, please follow these steps :
1. Download pretrained weights for yolo https://pjreddie.com/media/files/yolov3.weights and save them in the main directory with name yolov3.weights
2. Save the video you want to detect pedestrians on it in the main directory 
3. Launch the command : python video_detector.py --video [VideoName] --det det
4. A video with detections will be launch

## How to use it quickly

You have to use
'''bash
python main.py --video_path VIDEO_PATH --out_path OUT_PATH
'''

Where :
VIDEO_PATH : the path of your video, ex: 'video.avi', default='./data/video.avi/'
OUT_PATH : the directory of your outputs, ex: './output/', default='./output/'

## Other arguments

--dets : the directory path of your detections files if you have already existant ones, default=None
--deepsort_cfg : the path of the config json file for deepsort, default='.cfg/config.json'

## Final Report

https://www.overleaf.com/project/600aeb2b4d54a11fe0cb9d44 - report
https://www.overleaf.com/project/600dedb46770cf06f8384b83 - slides