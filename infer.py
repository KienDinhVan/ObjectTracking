import cv2
import numpy as np
import argparse
from model.sort.sort import Sort
from model.tracking_process import show_video
from model.tracking_process import process_video_sort
from model.tracking_process import process_video_deepsort
from model.tracking_process import process_video_bytetrack
import supervision as sv
from ultralytics import YOLO

model = YOLO('yolov10m.pt')

video_path = 'cars.mp4'
output_path = 'tracking_output.mp4'

print('Choose your tracking algorithm: \n'
      '1. Sort \n'
      '2. DeepSort \n'
      '3. ByteTrack')

idx = int(input('Index of algorithm: '))

if idx == 1:
      process_video_sort(video_path, output_path)
elif idx == 2:
      process_video_deepsort(video_path, output_path)
else:
      process_video_bytetrack(video_path, output_path)

show_video(output_path)