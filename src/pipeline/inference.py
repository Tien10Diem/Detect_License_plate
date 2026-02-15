import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
import argparse
import cv2

def agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_vehicle', default=r"", type=str, help='path to vehicle detection model')
    parser.add_argument('--weights_license_plate', default=r"", type=str, help='path to license plate detection model')
    parser.add_argument('--weights_character', default=r"", type=str, help='path to character detection model')

    return parser.parse_args()

class detect:
    def __init__(self, args):
        self.model_vehicle = YOLO(args.weights_vehicle)
        self.model_license_plate = YOLO(r"YoLo26_detect_LicensePlate\src\models\configs\yolo26m_custom_lp.yaml").load(args.weights_license_plate)
        self.model_character = YOLO(weights_path=args.weights_character)
        
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
        
        self.video_path = r"YoLo26_detect_LicensePlate\data\test_video\test_video_1.mp4"
        self.output_path = r"YoLo26_detect_LicensePlate\output\output_video_1.mp4"
    def interpolated(self, df):
        pass
    
    def crop(self, image, box):
        x1, y1, x2, y2 = map(int, box)
        return image[y1:y2, x1:x2]
    
    def detect_all(self):
        print(">>> Đang chạy Pipeline (Detect + Batch Inference + OCR)...")
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}...")
                
            rs_ve = self.detect_vehicle.track(source=frame, conf=0.45, verbose=False, persist=True, tracker="bytetrack.yaml")
            
            if rs_ve[0].boxes.id is not None:
                boxes = rs_ve[0].boxes.xyxy.cpu().numpy()
                track_ids = rs_ve[0].boxes.id.int().cpu().numpy()
                classes = rs_ve[0].boxes.cls.int().cpu().numpy()
                
                crop_batch = []      
                meta_info_batch = []  
                
                for box, track_id, cls in zip(boxes, track_ids, classes):
                    x1, y1, x2, y2 = box
                    self.track_history_vehicle.append([frame_count, track_id, x1, y1, x2, y2, cls])
                    
                    imgs_crop = self.crop_frame(frame, x1, y1, x2, y2)
                    
                    if imgs_crop.size > 0:
                        crop_batch.append(imgs_crop)
                        meta_info_batch.append((track_id, x1, y1, cls))

                if len(crop_batch) > 0:
                    rs_lp_batch = self.detect_license.predict(source=crop_batch, conf=0.25, verbose=False)

                    for i, rs_lp in enumerate(rs_lp_batch):
                        if len(rs_lp.boxes) > 0:
                            track_id, orig_x1, orig_y1, orig_cls = meta_info_batch[i]
                            
                            x1_lp, y1_lp, x2_lp, y2_lp = rs_lp.boxes[0].xyxy[0].cpu().numpy().astype(int)
                            
                            
                            crop_lp = self.crop_frame(crop_batch[i], x1_lp, y1_lp, x2_lp, y2_lp)
                            # cv2.imshow("crop_lp", crop_lp)
                            # cv2.waitKey(0)

                            ocr_text, ocr_conf = None, None
                            
                            ocr_text, ocr_conf = self.get_ocr_text(crop_lp)
                            if ocr_text is None:
                                ocr_text = ""
                                ocr_conf = 0.0
                            
                            real_x1 = orig_x1 + x1_lp
                            real_y1 = orig_y1 + y1_lp
                            real_x2 = orig_x1 + x2_lp
                            real_y2 = orig_y1 + y2_lp
                            
                            self.track_history_license.append([frame_count, track_id, real_x1, real_y1, real_x2, real_y2, orig_cls, ocr_text, ocr_conf])
                            
        cap.release()
        print(f"Đã xong {frame_count} frames.")