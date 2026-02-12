import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

class VehicleTrackerSystem:
    def __init__(self, model_path, input_video, output_video):
        """
        Khởi tạo hệ thống tracking với Modular Architecture
        """
        self.model = YOLO(model_path)
        self.input_video = input_video
        self.output_video = output_video
        self.track_history = []  # Nơi chứa dữ liệu thô
        
    def run_tracking(self):
        """
        Bước 1: Chạy YOLO + ByteTrack để lấy dữ liệu thô
        """
        print(">>> Đang chạy Tracking để thu thập dữ liệu...")
        cap = cv2.VideoCapture(self.input_video)
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            # Sử dụng ByteTrack (tracker='bytetrack.yaml')
            # persist=True là bắt buộc để giữ ID giữa các frame
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    x1, y1, x2, y2 = box
                    # Lưu dữ liệu: Frame hiện tại, ID xe, Tọa độ, Loại xe
                    self.track_history.append([frame_count, track_id, x1, y1, x2, y2, cls])
        
        cap.release()
        print(f">>> Đã thu thập xong dữ liệu của {frame_count} frames.")

    def process_interpolation(self):
        """
        Bước 2: Xử lý Nội suy (Interpolation) lấp đầy khoảng trống
        """
        print(">>> Đang xử lý nội suy (Smoothing)...")
        if not self.track_history:
            print("Không tìm thấy đối tượng nào để tracking.")
            return pd.DataFrame()

        # Chuyển list thành Pandas DataFrame để xử lý cho tiện
        df = pd.DataFrame(self.track_history, columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'class'])
        
        # Tạo bảng chứa kết quả sau khi nội suy
        interpolated_data = []

        # Lấy danh sách tất cả các xe (ID) đã xuất hiện
        unique_ids = df['track_id'].unique()

        for uid in unique_ids:
            # Lấy dữ liệu của riêng 1 chiếc xe
            car_data = df[df['track_id'] == uid].sort_values(by='frame')
            
            # Tạo lại index đầy đủ từ frame đầu đến frame cuối mà xe đó xuất hiện
            # Ví dụ: Xe xuất hiện ở frame 10 và 15, ta tạo các dòng trống 11, 12, 13, 14
            min_frame = car_data['frame'].min()
            max_frame = car_data['frame'].max()
            full_range = np.arange(min_frame, max_frame + 1)
            
            # Reindex để tạo khoảng trống
            car_data = car_data.set_index('frame').reindex(full_range)
            
            # --- CORE LOGIC: NỘI SUY TUYẾN TÍNH ---
            # Điền các giá trị NaN bằng phương pháp linear
            car_data[['x1', 'y1', 'x2', 'y2']] = car_data[['x1', 'y1', 'x2', 'y2']].interpolate(method='linear')
            
            # Điền ID và Class (Forward Fill - lấy giá trị trước đó điền vào)
            car_data['track_id'] = car_data['track_id'].ffill().bfill()
            car_data['class'] = car_data['class'].ffill().bfill()
            
            # Reset index để đưa 'frame' quay lại thành cột
            car_data = car_data.reset_index().rename(columns={'index': 'frame'})
            interpolated_data.append(car_data)
            
        # Gộp lại thành 1 bảng lớn
        final_df = pd.concat(interpolated_data)
        return final_df

    def render_output(self, dataframe):
        """
        Bước 3: Vẽ lại video từ dữ liệu đã nội suy
        """
        print(">>> Đang xuất video kết quả...")
        cap = cv2.VideoCapture(self.input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_idx += 1
            
            # Lấy dữ liệu của frame hiện tại từ bảng đã nội suy
            current_tracks = dataframe[dataframe['frame'] == frame_idx]
            
            for _, row in current_tracks.iterrows():
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                track_id = int(row['track_id'])
                cls = int(row['class'])
                
                # Vẽ Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Vẽ Label
                label = f"ID: {track_id}" 
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            out.write(frame)
            
        cap.release()
        out.release()
        print(f">>> Hoàn tất! Video đã lưu tại: {self.output_video}")

# --- CÁCH SỬ DỤNG ---
if __name__ == "__main__":
    # Thay đường dẫn của bạn vào đây
    processor = VehicleTrackerSystem(
        model_path=r"C:\Users\ASUS\Downloads\best_vehicle.pt",       # File model bạn đã train
        input_video=r"Video xe máy - xe hơi chạy trên đường - YouTube.mp4",    # Video đầu vào
        output_video='output_interpolated_test.mp4' # Video kết quả
    )
    
    # 1. Chạy tracking
    processor.run_tracking()
    
    # 2. Tính toán nội suy (Lấp frame trống)
    smooth_data = processor.process_interpolation()
    
    # 3. Xuất video
    processor.render_output(smooth_data)