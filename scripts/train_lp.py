import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.detect_custom_yaml import YoloDetector
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=24, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--imgsz', default=640, type=int, help='image size')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    custom_save_dir = str(ROOT / "weights" / "model_license_plate")
    
    model_license_plate = YoloDetector(r"src\models\configs\yolo26m_custom_lp.yaml", r'weights\yolo26m.pt')
    model_license_plate.model.train(
    data=r'data\Vietnam-license-plate-1\data.yaml',
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=0,
    lr0=1e-3,        
    optimizer='AdamW',
    name='license_plate',
    project=custom_save_dir,
)