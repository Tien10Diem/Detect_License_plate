import torch
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np


class inf:
    def __init__(self, input_video, output_video):
        self.input_video = input_video
        self.output_video = output_video
      
        self.detect_vehicle = YOLO(r"detect_vehicle\transfer_vehicle\best_vehicle.onnx")
        self.detect_license = YOLO(r'license_plate\runs\yolo26m_transfer\weights\best.onnx')
        
        self.track_history_vehicle = []
        self.track_history_license = []   
        
        self.model_detect_number = YOLO(r"recognize_number\best_recognize_number.pt")
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
    def crop_frame(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        return frame[y1:y2, x1:x2]
    
    # Nội suy cho xe
    def interpolate_tracks(self, dataset_Frame, gap=150):
        if len(dataset_Frame) == 0:
            print("None track history")
            return pd.DataFrame()
        
        df = pd.DataFrame(dataset_Frame, columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'class'])
        interpolated_data = []
        unique_ids = df['track_id'].unique()
        
        for uid in unique_ids:
            car_data = df[df['track_id'] == uid].sort_values(by='frame')
            full_range = np.arange(car_data['frame'].min(), car_data['frame'].max() + 1)
            
            car_data = car_data.set_index('frame').reindex(full_range)
            car_data[['x1', 'y1', 'x2', 'y2']] = car_data[['x1', 'y1', 'x2', 'y2']].interpolate(method='linear', limit=gap, limit_direction='both')
            
            car_data['track_id'] = car_data['track_id'].ffill().bfill()
            car_data['class'] = car_data['class'].ffill().bfill()
            
            car_data = car_data.dropna(subset=['x1', 'y1', 'x2', 'y2'])
            car_data = car_data.reset_index().rename(columns={'index': 'frame'})
            interpolated_data.append(car_data)
            
        final_df = pd.concat(interpolated_data)
        return final_df
    
    # Nội suy chi biển số
    def process_license_plates(self, dataset_lp, gap=90):
        if len(dataset_lp) == 0:
            return pd.DataFrame()
            
        # Khai báo 9 cột (có thêm Text và Score của OCR)
        df = pd.DataFrame(dataset_lp, columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'class', 'ocr_text', 'ocr_conf'])
        
        interpolated_data = []
        unique_ids = df['track_id'].unique()
        
        for uid in unique_ids:
            car_lp = df[df['track_id'] == uid].sort_values(by='frame')
            
            valid_texts = car_lp[(car_lp['ocr_text'] != "") & (car_lp['ocr_conf'] > 0.3)]['ocr_text']
            
            if not valid_texts.empty:
                best_text = valid_texts.mode()[0] 
            else:
                best_text = "Unknown"
                
            car_lp['final_text'] = best_text # Gán cứng Text đã bầu chọn
            
            full_range = np.arange(car_lp['frame'].min(), car_lp['frame'].max() + 1)
            car_lp = car_lp.set_index('frame').reindex(full_range)
            
            car_lp[['x1', 'y1', 'x2', 'y2']] = car_lp[['x1', 'y1', 'x2', 'y2']].interpolate(method='linear', limit=gap, limit_direction='both')
            car_lp['track_id'] = car_lp['track_id'].ffill().bfill()
            car_lp['final_text'] = car_lp['final_text'].ffill().bfill() 
            
            car_lp = car_lp.dropna(subset=['x1', 'y1', 'x2', 'y2'])
            car_lp = car_lp.reset_index().rename(columns={'index': 'frame'})
            
            interpolated_data.append(car_lp)
            
        return pd.concat(interpolated_data)
    
    def sort_ocr_results(self, boxes):
        """
        Hàm sắp xếp: Giữ nguyên logic gom dòng theo Y rồi sort theo X
        """
        if not boxes: return []
        
        # 1. Tính chiều cao trung bình
        avg_height = np.mean([b['h'] for b in boxes])
        
        # 2. Sắp xếp sơ bộ từ trên xuống dưới (theo Y)
        boxes.sort(key=lambda k: k['y'])
        
        lines = []
        current_line = [boxes[0]]
        
        # 3. Gom nhóm dòng
        for i in range(1, len(boxes)):
            box = boxes[i]
            prev_box = current_line[-1]
            
            # Nếu chênh lệch Y < 0.5 chiều cao -> Cùng 1 dòng
            if abs(box['y'] - prev_box['y']) < (avg_height * 0.5):
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
        lines.append(current_line)
        
        # 4. Sắp xếp trái -> phải trong từng dòng và merge lại
        final_sorted_boxes = []
        for line in lines:
            line.sort(key=lambda k: k['x'])
            final_sorted_boxes.extend(line)
            
        return final_sorted_boxes

    def get_ocr_text(self, frame):
        # Predict
        results = self.model_detect_number.predict(source=frame, conf=0.15, verbose=False)
        
        if len(results[0].boxes) > 0: 
            pos = []              
            conf_list = [] 

            for res in results[0].boxes:
                # Lấy toạ độ
                x1, y1, x2, y2 = res.xyxy[0].cpu().numpy().astype(int)
                
                # Lấy ID class và Confidence
                class_id = int(res.cls[0].cpu().numpy())
                conf = float(res.conf[0].cpu().numpy())
                
                # BƯỚC 1: Lấy tên class do model dự đoán (ví dụ: '0', '29')
                raw_class_name = self.model_detect_number.names[class_id] 
                
                # BƯỚC 2: Map qua từ điển của bạn
                # Dùng hàm .get() để tránh báo lỗi nếu model văng ra một class lạ
                mapped_char = self.char_mapping.get(raw_class_name, raw_class_name)
                
                dic = {
                    'char': mapped_char, # Lưu ký tự đã được map chuẩn
                    'conf': conf,
                    'x': x1,
                    'y': y1,
                    'w': x2 - x1,
                    'h': y2 - y1
                }
                pos.append(dic)
                conf_list.append(conf)
            
            # Sắp xếp
            pos_sorted = self.sort_ocr_results(pos)
            
            # Tạo text final
            final_text = ''.join([item['char'] for item in pos_sorted])
            
            # Tính score trung bình
            avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0
            
            return final_text, avg_conf
        else:
            return "", 0.0
        
    def detect(self):
        print(">>> Đang chạy Pipeline (Detect + Batch Inference + OCR)...")
        cap = cv2.VideoCapture(self.input_video)
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

    def draw_tracks(self, df_vehicle, df_license):
        print("Đang xuất Video kết quả...")
        cap = cv2.VideoCapture(self.input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_idx = 0
        
        veh_grouped = df_vehicle.groupby('frame') if not df_vehicle.empty else None
        lp_grouped = df_license.groupby('frame') if not df_license.empty else None
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_idx += 1
            
            curr_vehicles = veh_grouped.get_group(frame_idx) if veh_grouped is not None and frame_idx in veh_grouped.groups else pd.DataFrame()
            curr_lps = lp_grouped.get_group(frame_idx) if lp_grouped is not None and frame_idx in lp_grouped.groups else pd.DataFrame()
            
            for _, row in curr_vehicles.iterrows():
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                track_id = int(row['track_id'])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID: {track_id}" 
                cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
            for _, row in curr_lps.iterrows():
                x1_lp, y1_lp, x2_lp, y2_lp = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                
                plate_text = row['final_text']
                
                cv2.rectangle(frame, (x1_lp, y1_lp), (x2_lp, y2_lp), (0, 0, 255), 2)
                
                cv2.putText(frame, plate_text, (x1_lp, max(0, y1_lp - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
            out.write(frame)
            
        cap.release()
        out.release()
        print(f"Video lưu tại: {self.output_video}")
    
    def run(self):
        self.detect()
        
        print("Nội suy dữ liệu Xe...")
        smooth_data = self.interpolate_tracks(self.track_history_vehicle, gap=150)
        
        print(" Xử lý Bầu chọn OCR & Nội suy dữ liệu Biển số...")
        
        smooth_data_lp = self.process_license_plates(self.track_history_license, gap=90)
        
        self.draw_tracks(smooth_data, smooth_data_lp)
        
if __name__ == "__main__":
    in_vid = r"video_cut.mp4"
    out_vid = 'interpolated_test_ocr.mp4'
    
    app = inf(input_video=in_vid, output_video=out_vid)
    app.run()