# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:35:38 2023

@author: rebel
"""

import cv2
import torch
import numpy as np

# Modelul pentru detectare de persoana/obiecte (YOLO)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define a video capture object
vid = cv2.VideoCapture(0)

#Etichetele 
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']

while True:
    # Ret este o variabila de tip bool returnata de functia vidread()
    #frame_inin este cadrul trimis de catre functia vid.read()
    ret, frame_ini = vid.read()
    
    
    
    if(ret==False):
        print("Nu a pornit camera")
    
    #face reverse pentru a fi mai natural(1 reprezinta filp orizonstal)
    frame=cv2.flip(frame_ini,1)
    
    # Perform object detection on the frame (assuming frame is already on the GPU)
    results = model(frame)  # Send the frame to the GPU and run the model

    # Get bounding box and class info (no need to move to CPU)
    bbox = results.pred[0][:, :4].cpu().numpy()
    conf = results.pred[0][:, 4].cpu().numpy()
    class_ids = results.pred[0][:, 5].cpu().numpy()


    # Filter results based on confidence threshold (adjust as needed)
    mask = conf > 0.7
    bbox = bbox[mask]
    class_ids = class_ids[mask].astype(int)

    for i in range(len(bbox)):
        
        x1, y1, x2, y2 = map(int, bbox[i])
        class_id = class_ids[i]
        class_name = class_names[class_id]
        label = f'{class_name}{conf[i]:.2f}'
        color = (255, 0, 0)  # BGR color code (green)

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Apasarea butonul q sparge bucla while
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, release the video capture object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()