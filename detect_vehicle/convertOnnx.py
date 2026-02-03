from ultralytics import YOLO

def main():
    model_path = r"detect_vehicle\transfer_vehicle\best.pt"
    
    model = YOLO(model_path)

    path = model.export(
        format="onnx",      # Onnx
        imgsz=640,          
        opset=12,           # version of onnx
        simplify=True       # Tự động rút gọn đồ thị ONNX cho nhẹ
    )

if __name__ == "__main__":
    main()