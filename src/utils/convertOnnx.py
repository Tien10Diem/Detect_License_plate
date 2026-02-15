from ultralytics import YOLO
import argparse
# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parents[2]))

def agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=r"", type=str, help='path to model')

    return parser.parse_args()
def main():
    args = agrs()
    
    model = YOLO(args.path)

    path = model.export(
        format="onnx",      # Onnx
        imgsz=640,          
        opset=12,           # version of onnx
        simplify=True,      # Tự động rút gọn đồ thị ONNX cho nhẹ
        dynamic=True,       # Kích thước đầu vào động
        # half = True,  # giảm 1 nửa số bit
    )

if __name__ == "__main__":
    main()