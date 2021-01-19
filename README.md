# DeepSort_tracking
Centralesupélec Deep Learning Course project on multi object tracking with DeepSort

## Yolo
To detect pedestrians in a video, please follow these steps :
1. Download pretrained weights for yolo https://pjreddie.com/media/files/yolov3.weights and save them in the main directory with name yolov3.weights
2. Save the video you want to detect pedestrians on it in the main directory 
3. Launch the command : python video_detector.py --video [VideoName] --det det
4. A video with detections will be launch
