from ultralytics import YOLO
from CBAM import CBAM, ChannelAttention, SpatialAttention
from ultralytics.nn import tasks
tasks.CBAM = CBAM

def main():
    model_path = r"license_plate\runs\yolo26m_transfer\weights\best.pt"
    
    model = YOLO(model_path)

    path = model.export(
        format="onnx",      # Onnx
        imgsz=640,          
        opset=12,           # version of onnx
        simplify=True,     
        # half = True,  # giảm 1 nửa số bit
    )

if __name__ == "__main__":
    main()