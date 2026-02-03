from ultralytics import YOLO
import argparse


def agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default= 50, type=int, help='batch size')
    parser.add_argument('--epochs', default= 50, type=int, help='epochs')
    parser.add_argument('--imgsz', default= 640, type=int, help='image size')
    
    return parser.parse_args()



if __name__ == "__main__":
    # Load model
    model = YOLO("yolo26m.pt") 
    
    args = agrs()
    
    # transfer learning 
    results = model.train(
        data="detect_vehicle\Vehicle-1\data.yaml", 
        epochs=args.epochs,
        batch=args.batch,       
        imgsz=args.imgsz, 
        workers=2,
        device='0'
)