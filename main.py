import csv
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from functions import get_car, read_license_plate, write_csv, interpolate_bounding_boxes, process_video
from sort.sort import *

results = {}
mot_tracker = Sort()

# Loading Model
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./model/train2/weights/best.pt')


# Loading videos
cap = cv2.VideoCapture('./video/sample.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")

vehicles = [2, 3, 5, 7]

# Reading Frames
frame_nmr =-1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # Vehicle Detection
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Vehicle tracking
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Plate Detector
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assigning License plates to vehicles
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # Crop Plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Processing crop plate
                license_plate_crp_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crp_gray, 64, 255, cv2.THRESH_BINARY_INV)

                """
                cv2.imshow('original', crop_plate)
                cv2.imshow('threshold', crop_plate_gray_thresh)
                cv2.waitKey(0)
                """

                # read plate number
                license_plate_text, license_plate_txt_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                 'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_txt_score}}


# Results
write_csv(results, './results/test.csv')

# Load the CSV file
with open('./results/test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('./results/test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

# Define paths
results_path = './results/test_interpolated.csv'
video_path = './video/sample.mp4'
output_path = './out.mp4'

# Load results
results = pd.read_csv(results_path)

# Process the video with the provided data
process_video(results, video_path, output_path)
