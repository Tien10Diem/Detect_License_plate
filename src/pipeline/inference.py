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
    parser.add_argument('--output_path', default=r"sample\output\output.mp4", type=str, help='path to output video')
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
        results = self.model_license_plate.predict(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False) # verbose=False để tắt log chi tiết 
        return results
    
    def sort_ocr_results(self, rs):
        if len(rs[0].boxes) == 0:
            return ""
        
        data = []
        for res in rs[0].boxes:
            # Dùng xyxy giống teacher để lấy chuẩn mép trái trên
            x1, y1, x2, y2 = res.xyxy[0].cpu().numpy().astype(int)
            class_id = int(res.cls[0].item())
            
            # Lấy tên class gốc và map chuẩn giống teacher
            raw_class_name = self.model_character.names[class_id]
            char = self.char_mapping.get(raw_class_name, raw_class_name)
            
            h = y2 - y1
            data.append({'char': char, 'x': x1, 'y': y1, 'h': h})
            
        # Tính chiều cao trung bình
        avg_h = np.mean([item['h'] for item in data])
        
        # Sắp xếp sơ bộ từ trên xuống dưới
        data.sort(key=lambda item: item['y']) 
        
        lines = []
        current_line = [data[0]]
        
        # Gom dòng động (chống biển số nghiêng)
        for i in range(1, len(data)):
            box = data[i]
            prev_box = current_line[-1]
            
            if abs(box['y'] - prev_box['y']) < (avg_h * 0.5):
                current_line.append(box)
            else:             
                lines.append(current_line)
                current_line = [box]
        lines.append(current_line)
        
        # Sắp xếp trái -> phải cho từng dòng
        text = ""
        for line in lines:
            line.sort(key=lambda item: item['x'])    
            text += ''.join([item['char'] for item in line]) 
        
        return text
    
    def detect_character(self, frame, conf_threshold=0.5, iou_threshold=0.5):
        results = self.model_character.predict(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        text= self.sort_ocr_results(results)

        return text
    
    def crop_license_plate(self, frame, xyxy):
        if len(xyxy) == 0:
            return None
        x1, y1, x2, y2 = map(int, xyxy)
        return frame[y1:y2, x1:x2]  
    
    def interpolated_vehicle(self, gap = 150):
        df = pd.DataFrame(self.track_history_vehicle, columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
        
        # Lấy id độc nhất
        track_ids = df['track_id'].unique()
       
        interpolated_data = []
        for track_id in track_ids:
            track_data = df[df['track_id'] == track_id].sort_values(by='frame')
            full_frames = range(track_data['frame'].min(), track_data['frame'].max() + 1)
            
            # Biến đổi frame thành index để dễ dàng nội suy
            track_data = track_data.set_index('frame').reindex(full_frames) # reindex để thêm đủ các frame liên tiếp (điền NaN cho các frame thiếu)
            track_data[['x1', 'y1', 'x2', 'y2']] = track_data[['x1', 'y1', 'x2', 'y2']].interpolate(method='linear', limit= gap) # nội suy tuyến tính với giới hạn gap
            track_data['track_id'] = track_id  
            
            # Biến đổi index thành cột 'frame'
            track_data = track_data.reset_index() 
            track_data = track_data.rename(columns={'index': 'frame'})
            
            track_data = track_data.dropna(subset=['x1', 'y1', 'x2', 'y2']) # loại bỏ các frame vẫn còn NaN sau nội suy (nằm ngoài giới hạn gap)
                        
            interpolated_data.append(track_data)
        
        interpolated_df = pd.concat(interpolated_data, ignore_index=True)
        return interpolated_df 

    def interpolated_license(self, gap= 150):
        df = pd.DataFrame(self.track_history_license, columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'text'])
        
        # xử lý text để nội suy 
        df['text'] = df['text'].replace('', np.nan) # thay thế text rỗng bằng NaN để nội suy
            
        # Lấy id độc nhất
        track_ids = df['track_id'].unique()
       
        interpolated_data = []
        
        for track_id in track_ids:
            
            track_data = df[df['track_id'] == track_id].sort_values(by='frame')
            full_frames = range(track_data['frame'].min(), track_data['frame'].max() + 1)
            
            # Xử lý text trước để nội suy
            if not track_data['text'].isnull().all():  # chỉ nội suy nếu có ít nhất một text không rỗng
                best_text = track_data['text'].mode()[0]  # thay thế NaN bằng text phổ biến nhất (mode) để giữ nguyên text trong quá trình nội suy
            else:
                best_text = "unknown"
            
            print(best_text)
            
            # Biến đổi frame thành index để dễ dàng nội suy
            track_data = track_data.set_index('frame').reindex(full_frames) # reindex để thêm đủ các frame liên tiếp (điền NaN cho các frame thiếu)
            track_data[['x1', 'y1', 'x2', 'y2']] = track_data[['x1', 'y1', 'x2', 'y2']].interpolate(method='linear', limit= gap) # nội suy tuyến tính với giới hạn gap
            track_data['track_id'] = track_id  
            
            # Biến đổi index thành cột 'frame'
            track_data = track_data.reset_index() 
            track_data = track_data.rename(columns={'index': 'frame'})
            
            track_data = track_data.dropna(subset=['x1', 'y1', 'x2', 'y2']) # loại bỏ các frame vẫn còn NaN sau nội suy (nằm ngoài giới hạn gap)
            
            track_data['text'] = best_text  # gán lại text đã xử lý cho tất cả các frame của track_id này
            
            interpolated_data.append(track_data)
        
        interpolated_df = pd.concat(interpolated_data, ignore_index=True)
        return interpolated_df
    
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
            rs = vehicle_results[0]
            if rs.boxes.id is not None:
                boxes = rs.boxes.xyxy.cpu().numpy()
                track_ids = rs.boxes.id.int().cpu().numpy()
                classes = rs.boxes.cls.int().cpu().numpy()

                # Bọc lại trong zip để xử lý đồng thời box, track_id và class
                for box, track_id, cls in zip(boxes, track_ids, classes):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    self.track_history_vehicle.append((frame_count, track_id, x1, y1, x2, y2))
                    
                    # Crop ảnh xe
                    frame_crop =self.crop_license_plate(frame, box)
                    license_plate_results = self.detect_license_plate(frame_crop)
                    
                    boxes_lp = license_plate_results[0].boxes.xyxy.cpu().numpy()
                    if len(boxes_lp) > 0:
                        # Crop ảnh biển số
                        boxes_lp = boxes_lp[0] # output luôn có dạng [[...]]
                        crop_lp = self.crop_license_plate(frame_crop, boxes_lp) 
                    
                        # Nhận diện ký tự trên biển số
                        text = self.detect_character(crop_lp, conf_threshold=0.15, iou_threshold=0.5)
                        
                        # Cộng lại tọa độ cho ảnh
                        x1_lp = x1 + int(boxes_lp[0])
                        y1_lp = y1 + int(boxes_lp[1])
                        x2_lp = x1 + int(boxes_lp[2])
                        y2_lp = y1 + int(boxes_lp[3])
                        
                        self.track_history_license.append((frame_count, track_id, x1_lp, y1_lp, x2_lp, y2_lp, text))
        cap.release()
        
        print(f"Đã xong {frame_count} frames.")
    
    def run(self):
        self.process_video()
        interpolated_vehicle_df = self.interpolated_vehicle()
        interpolated_license_df = self.interpolated_license()
        
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            # Vẽ bounding box xe
            vehicle_boxes = interpolated_vehicle_df[interpolated_vehicle_df['frame'] == frame_count][['x1', 'y1', 'x2', 'y2']].values # values để lấy về dạng numpy array
            for box in vehicle_boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ bounding box biển số và text
            license_boxes = interpolated_license_df[interpolated_license_df['frame'] == frame_count][['x1', 'y1', 'x2', 'y2', 'text']].values # values để lấy về dạng numpy array
            for box in license_boxes:
                x1, y1, x2, y2, text = box
                x1_lp, y1_lp, x2_lp, y2_lp = int(x1), int(y1), int(x2), int(y2)
                
                cv2.rectangle(frame, (x1_lp, y1_lp), (x2_lp, y2_lp), (0, 0, 255), 2)
                cv2.putText(frame, str(text), (x1_lp, max(0, y1_lp - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"Video lưu tại: {self.output_path}")
        
if __name__ == "__main__":
    args = agrs()
    app = detect(args)
    app.run()
    