import onnx
from onnxconverter_common import float16

def convert_to_fp16(input_model_path, output_model_path):
    print(f"⏳ Dang chuyen doi sang FP16: {input_model_path} ...")
    
    try:
        # 1. Load model gốc (32-bit)
        model = onnx.load(input_model_path)
        
        # 2. Convert sang Float16
        # keep_io_types=True: Giữ đầu vào/ra là Float32 để code inference không bị lỗi type
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        
        # 3. Lưu file mới
        onnx.save(model_fp16, output_model_path)
        print(f"✅ Xong! Da luu tai: {output_model_path}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

# --- THỰC HIỆN ---

# 1. Model detect xe
convert_to_fp16(r"detect_vehicle\transfer_vehicle\best.onnx", 
                r"detect_vehicle\transfer_vehicle\best_fp16.onnx")

# 2. Model detect bien so
convert_to_fp16(r"license_plate\runs\yolo26m_transfer\weights\best.onnx", 
                r"license_plate\runs\yolo26m_transfer\weights\best_fp16.onnx")