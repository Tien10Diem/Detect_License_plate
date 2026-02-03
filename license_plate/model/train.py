from ultralytics import YOLO
from ultralytics.nn import tasks
from CBAM import CBAM
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default= 50, type=int, help='batch size')
    parser.add_argument('--epochs', default= 50, type=int, help='epochs')
    parser.add_argument('--imgsz', default= 640, type=int, help='image size')
    
    return parser.parse_args()


if __name__ == "__main__":

    tasks.CBAM = CBAM
    custom_yaml = r"license_plate\yolo26m_custom.yaml"
    model = YOLO(custom_yaml).load('yolo26m.pt')

    args = args()

    results = model.train(
        data='license_plate\dataset.yaml',
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        device=0,
        freeze=0,        
        lr0=0.001,       
        optimizer='AdamW',
        project=r'detect_license_plate/runs',
        name='yolo26m_transfer',
    )