import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
import argparse
import cv2
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.models.yolo_builder 


def agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_vehicle', default=r"weights\model_vehicle\vehicle_detector.onnx", type=str, help='path to vehicle detection model')
    parser.add_argument('--weights_license_plate', default=r"weights\model_license_plate\plate_detector_best.onnx", type=str, help='path to license plate detection model')
    parser.add_argument('--weights_character', default=r"weights\model_numbers\best_recognize_number.onnx", type=str, help='path to character detection model')
    parser.add_argument('--video_path', default=r"sample\raw_video\video_cut.mp4", type=str, help='path to input video')
    parser.add_argument('--output_path', default=r"sample\output\output1.mp4", type=str, help='path to output video')
    return parser.parse_args()

class detect:
    def __init__(self, args):
        self.model_vehicle = YOLO(args.weights_vehicle)
        self.model_license_plate = YOLO(args.weights_license_plate)
        self.model_character = YOLO(args.weights_character)
        
        self.char_mapping = {
            '0': '1', '1': '2', '2': '3', '3': '4', '4': '5', 
            '5': '6', '6': '7', '7': '8', '8': '9', '9': 'A',
            '10': 'B', '11': 'C', '12': 'D', '13': 'E', '14': 'F',
            '15': 'G', '16': 'H', '17': 'K', '18': 'L', '19': 'M',
            '20': 'N', '21': 'P', '22': 'Q', '23': 'R', '24': 'S', 
            '25': 'T', '26': 'X', '27': 'V', '28': 'W', '29': '0',
            '30': 'Y', '31': 'Z',
            '32': '0', '33': '0' 
        }
        self.track_history_vehicle = []
        self.track_history_license = []   
        
        self.video_path = args.video_path
        self.output_path = args.output_path
    
    def detect_vehicle(self, frame, conf_threshold=0.45, iou_threshold=0.5):
        results = self.model_vehicle.track(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False, persist=True, tracker="bytetrack.yaml")
        return results
    
    def detect_license_plate(self, frame, conf_threshold=0.5, iou_threshold=0.5):
        results = self.model_license_plate.predict(source=frame, conf=conf_threshold, iou=iou_threshold)
        return results
    
    def sort_ocr_results(self, rs):
        
        sum_h= 0
        x_y = []
        sum_x = 0
        cls = []
        if len(rs) == 0:
            return []
        for res in rs[0].boxes:
            x,y,w,h = res.xywh[0].cpu().numpy().astype(int)
            clses = res.cls.cpu().numpy().astype(int)
            cls.append(self.char_mapping.get(str(clses), ''))
            sum_x += x
            sum_h += h
            x_y.append((x,y))
        avg_h = sum_h / len(rs[0].boxes)*0.5
        x_y.sort(key=lambda item: item[1])
        y0 = x_y[0][1]
        avg_x = sum_x / len(rs[0].boxes)
        
        top = {}
        bottom = {}
        for i in range(len(x_y)):
            if abs(x_y[i][1] - y0) < avg_h:
                top[cls[i]] = x_y[i][0]
            else:             
                bottom[cls[i]] = x_y[i][0]
        
        top_sorted = sorted(top.items(), key=lambda item: item[1])
        bottom_sorted = sorted(bottom.items(), key=lambda item: item[1])
        text = ''.join([item[0] for item in top_sorted]) + ''.join([item[0] for item in bottom_sorted])    
        
        return text
    
    def detect_character(self, frame, conf_threshold=0.5, iou_threshold=0.5):
        results = self.model_character.predict(source=frame, conf=conf_threshold, iou=iou_threshold)
        
        text= self.sort_ocr_results(results)

        return text
    
    def crop_license_plate(self, frame, xyxy):
        if len(xyxy) == 0:
            return None
        x1, y1, x2, y2 = map(int, xyxy)
        return frame[y1:y2, x1:x2]  
    
    def interpolated_vehicle(self):
        df = pd.DataFrame(self.track_history_vehicle, columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
        ranges_frame = range(df['frame'].min(), df['frame'].max() + 1)
        
        # Lấy id độc nhất
        track_ids = df['track_id'].unique()
       
        interpolated_data = []
        for track_id in track_ids:
            track_data = df[df['track_id'] == track_id].sort_values(by='frame')
    
    def interpolated_license(self):
        pass
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}...")
                
            vehicle_results = self.detect_vehicle(frame)
            for rs in vehicle_results[0]:
                if rs.boxes.id is not None:
                    boxes = rs.boxes.xyxy.cpu().numpy()
                    track_ids = rs.boxes.id.int().cpu().numpy()
                    classes = rs.boxes.cls.int().cpu().numpy()

                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        self.track_history_vehicle.append((frame_count, track_id, x1, y1, x2, y2))
                        
                        # Crop ảnh xe
                        frame_crop =self.crop_license_plate(frame, box)
                        license_plate_results = self.detect_license_plate(frame_crop)
                        
                        boxes_lp = license_plate_results[0].boxes.xyxy.cpu().numpy()
                        if len(boxes_lp) > 0:
                            # Crop ảnh biển số
                            boxes_lp = boxes_lp[0] # output luôn có dạng [[...]]
                            crop_lp = self.crop_license_plate(frame_crop, boxes_lp) 
                            cv2.rectangle(frame, (boxes_lp[0][0], boxes_lp[0][1]), (boxes_lp[0][2], boxes_lp[0][3]), (255, 0, 0), 2)
                            
                            # Nhận diện ký tự trên biển số
                            text = self.detect_character(crop_lp)
                            
                            self.track_history_license.append((frame_count, track_id, boxes_lp[0][0], boxes_lp[0][1], boxes_lp[0][2], boxes_lp[0][3], text))
                            cv2.putText(frame, text, (boxes_lp[0][0], boxes_lp[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
            
        cap.release()
    