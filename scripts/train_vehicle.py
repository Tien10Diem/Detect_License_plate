import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from src.models.detect_normal import DetectNormal 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=50, type=int, help='batch size')
    parser.add_argument('--epochs', default=32, type=int, help='epochs')
    parser.add_argument('--imgsz', default=640, type=int, help='image size')
    
    parser.add_argument('--weights', default=r'weights\yolo26m.pt', type=str, help='Đường dẫn file .pt (Dùng last.pt nếu resume)')
    parser.add_argument('--resume', action='store_true', help='Kích hoạt chế độ tiếp tục huấn luyện')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_1 = DetectNormal(weights_path=args.weights)
    
    ROOT = Path(__file__).resolve().parents[1]
    custom_save_dir = str(ROOT / "weights" / "model_vehicle")
    # custom_save_dir = r"weights\model_vehicle"
    
    model_1.model.train(
    data=r"data\YOLOv8-2-1\data.yaml", 
    epochs=args.epochs,
    batch=args.batch,       
    imgsz=args.imgsz, 
    workers=2,
    device='0',
    degrees=15.0, 
    shear=2.5,     
    erasing=0.4,   
    mosaic=1.0,
    project=custom_save_dir,
    name="vehicle",
    )