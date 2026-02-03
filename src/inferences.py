import torch
from ultralytics import YOLO
import cv2
from fast_plate_ocr import LicensePlateRecognizer
import os
import re
import argparse
import onnxruntime as ort
from paddleocr import PaddleOCR

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle', '-v', default= r'detect_vehicle\transfer_vehicle\best.onnx', type=str, help='model path')
    parser.add_argument('--license_plate', '-l', default= r'license_plate\runs\yolo26m_transfer\weights\best.onnx', type=str, help='model path')
    parser.add_argument('--ocr', '-o', default= r'cct-xs-v1-global-model', type=str, help='model path')
    parser.add_argument('--path', '-p', default= r'Video xe máy - xe hơi chạy trên đường - YouTube.mp4', type=str, help='video path')
    parser.add_argument('--save_path', '-s', default= r'inf', type=str, help='video path')
    
    return parser.parse_args()

class detect:
    def __init__(self, vehicle, license_plate, ocr, path, save_path):
        if torch.cuda.is_available():
            print("PyTorch đang dùng GPU")
        else:
            print("PyTorch đang dùng CPU")
            
        print(f"ONNX Providers: {ort.get_available_providers()}")
        
        self.model_vehicle = YOLO(vehicle, task='detect')
        self.model_lp = YOLO(license_plate, task='detect')
        self.ocr_model = ""
        self.path = path
        self.save_path = save_path
        
        if not os.path.exists(save_path): os.makedirs(save_path)
        
    def crop(self, frame, x1, y1, x2, y2):

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]
    
    def save(self, frame, lp, path):
        cv2.imwrite(f'{path}/{lp}.jpg', frame)
                    
        
    def detect_vehicle(self, frame):
        results = self.model_vehicle.predict(source=frame, conf=0.25, verbose=False)
        return results
    
    def ocr(self, plate_img):
        pass
        
    def detect_license_plate(self, frame):
        results = self.model_lp.predict(source=frame, conf=0.25, verbose=False)
        
        if len(results[0].boxes) == 0:
            return None, None
        
        x1, y1, x2, y2 = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    
        crop = self.crop(frame, x1, y1, x2, y2)
        # cv2.imshow("crop", crop)
        # cv2.waitKey(0)
        
        return crop, (x1, y1, x2, y2)
    
    def detect_all(self, frame):
        results = self.detect_vehicle(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                crop = self.crop(result.orig_img, x1, y1, x2, y2)
                plate_crop, lxy= self.detect_license_plate(crop)
    
                if plate_crop is not None and plate_crop.size > 0:
                    lx1, ly1, lx2, ly2 = lxy
                    lx1 += x1
                    ly1 += y1
                    lx2 += x1
                    ly2 += y1
                    
                    # text = self.ocr(plate_crop)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
                    # cv2.putText(frame, text, (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
    def run(self):
        
        if self.path.endswith(('.mp4', '.avi', '.mkv')):
            cap = cv2.VideoCapture(self.path)
            
            skip_frame = 0
            while True:
                flag, frame = cap.read()
                
                if not flag:
                    break
                if skip_frame == 0:
                    self.detect_all(frame)
                    skip_frame = 0
                    
                else:
                    skip_frame -= 1
                
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            cap.release()
            cv2.destroyAllWindows()
                
        elif self.path.endswith(('.jpg', '.png', '.jpeg')):
            frame = cv2.imread(self.path)
            self.detect_all(frame)
            
            # Với ảnh thì waitKey(0) để dừng lại xem
            cv2.imshow("Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    args = args()
    vehicle = args.vehicle
    license_plate = args.license_plate
    ocr = args.ocr
    path = args.path
    save_path = args.save_path
    
    detect = detect(vehicle, license_plate, ocr, path, save_path)
    detect.run()