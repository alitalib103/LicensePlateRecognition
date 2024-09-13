import csv
import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from functions import (
    get_car,
    read_license_plate,
    write_csv,
    interpolate_bounding_boxes,
    process_video,
)
from sort.sort import Sort
import tempfile

# Initialize models and tracker
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./model/train2/weights/best.pt')
mot_tracker = Sort()

# Streamlit App Configuration
st.title("Automatic License Plate Recognition System")
st.write("Upload a video to detect and recognize vehicle license plates.")

# Upload Video File
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Processing Logic
if uploaded_file is not None:
    # Save uploaded video to a temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing video, please wait...")

    # Load video
    cap = cv2.VideoCapture(temp_video_path)

    # Check if video loaded successfully
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
    else:
        # Define temporary paths
        temp_csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        temp_interpolated_csv_path = tempfile.NamedTemporaryFile(delete=False, suffix="_interpolated.csv").name
        temp_output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Vehicle classes from COCO dataset
        vehicles = [2, 3, 5, 7]

        # Initialize variables
        results = {}
        frame_nmr = -1
        ret = True

        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                results[frame_nmr] = {}

                # Vehicle Detection
                detections = coco_model(frame)[0]
                detections_ = [
                    [x1, y1, x2, y2, score]
                    for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist()
                    if int(class_id) in vehicles
                ]

                # Vehicle tracking
                track_ids = mot_tracker.update(np.asarray(detections_))

                # Plate Detection
                license_plates = license_plate_detector(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    if car_id != -1:
                        # Crop the detected license plate area
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        license_plate_crp_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(
                            license_plate_crp_gray, 64, 255, cv2.THRESH_BINARY_INV
                        )

                        # Read license plate number
                        license_plate_text, license_plate_txt_score = read_license_plate(license_plate_crop_thresh)

                        if license_plate_text is not None:
                            results[frame_nmr][car_id] = {
                                "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                                "license_plate": {
                                    "bbox": [x1, y1, x2, y2],
                                    "text": license_plate_text,
                                    "bbox_score": score,
                                    "text_score": license_plate_txt_score,
                                },
                            }

        # Save results to CSV
        write_csv(results, temp_csv_path)

        # Interpolate missing bounding boxes
        with open(temp_csv_path, "r") as file:
            reader = csv.DictReader(file)
            data = list(reader)

        interpolated_data = interpolate_bounding_boxes(data)

        # Write interpolated data to a new CSV file
        header = [
            "frame_nmr",
            "car_id",
            "car_bbox",
            "license_plate_bbox",
            "license_plate_bbox_score",
            "license_number",
            "license_number_score",
        ]
        with open(temp_interpolated_csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(interpolated_data)

        # Load the results into DataFrame
        results_df = pd.read_csv(temp_interpolated_csv_path)

        # Process the video with results and save it
        process_video(results_df, temp_video_path, temp_output_video_path)

        # Provide a download link for the processed video
        with open(temp_output_video_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )