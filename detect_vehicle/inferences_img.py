from ultralytics import YOLO
import argparse
import cv2

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default= r'detect_vehicle\transfer_vehicle\best.pt', type=str, help='model path')
    parser.add_argument('--path', '-p', default= r'detect_vehicle\Vehicle-1\test\images\0000144_01201_d_0000001_jpg.rf.37f196495046f0c812d791cfa204c102.jpg', type=str, help='video path')
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