#!/usr/bin/env python
import os
img_files=set([x[:-4] for x in os.listdir("/workspace/src/detections/yolov8/3/images")])
vis_files=set([x[:-4] for x in os.listdir("/workspace/src/detections/yolov8/3/vis")])
for file in list(img_files-vis_files):
    os.remove(f"/workspace/src/detections/yolov8/3/labels/{file}.txt")
    os.remove(f"/workspace/src/detections/yolov8/3/images/{file}.jpg")
