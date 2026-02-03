from ultralytics import YOLO
import argparse
import cv2
from model.CBAM import CBAM, ChannelAttention, SpatialAttention
from ultralytics.nn import tasks

tasks.CBAM = CBAM

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default= r'license_plate\runs\yolo26m_transfer\weights\best.pt', type=str, help='model path')
    parser.add_argument('--path', '-p', default= r'license_plate\Vietnam-license-plate-1\test\images\CarLongPlate20_jpg.rf.9218573c6249c89f9d361c4925026be0.jpg', type=str, help='video path')
    parser.add_argument('--conf', '-c', default= 0.25, type=float, help='confidence')
    
    return parser.parse_args()

def predict(model_best, conf, img):
    model = YOLO(model_best)
    results = model(img, imgsz=640, conf = conf)
    result_img = results[0].plot()
    cv2.imshow("output",result_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    args = args()
    model = args.model
    path = args.path
    conf = args.conf
    predict(model, conf, path)