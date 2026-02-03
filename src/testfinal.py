import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models

# Tắt log rác
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- CẤU HÌNH ---
IMG_WIDTH = 200
IMG_HEIGHT = 50
CHAR_LIST = "0123456789ABCDEFGHKLMNPSTUVXYZ-. JQORĐI"

char_to_num = layers.StringLookup(vocabulary=list(CHAR_LIST), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
# --- 1. THÊM HÀM SỬA LỖI VÀO ĐẦU FILE EVALUATE.PY ---
def autocorrect_plate(text):
    text = text.replace('[UNK]', '').replace('_', '').replace('-', '').replace('.', '').strip()
    
    # Từ điển sửa lỗi
    dict_char_to_num = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'J': '3', 'L': '1', 'S': '5', 'B': '8', 'Z': '7', 'A': '4', 'G': '6'}
    dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '7': 'T', '6': 'G'}

    chars = list(text)
    new_chars = []

    for i, c in enumerate(chars):
        # 2 ký tự đầu là SỐ (Mã tỉnh)
        if i < 2:
            new_chars.append(dict_char_to_num.get(c, c))
        # Ký tự thứ 3 là CHỮ (Series)
        elif i == 2:
            # Nếu là số thì cố ép về chữ, không thì giữ nguyên
            if c.isdigit():
                new_chars.append(dict_num_to_char.get(c, c))
            else:
                new_chars.append(c)
        # Các ký tự sau là SỐ
        elif i > 2:
            new_chars.append(dict_char_to_num.get(c, c))
            
    return "".join(new_chars)

# --- 2. SỬA ĐOẠN VÒNG LẶP TRONG HÀM evaluate ---
    # ...
    # for i in range(total):
        # ... (đoạn dự đoán giữ nguyên)
        # pred_text = decode_batch_predictions(probs)[0]
        
        # 🔥 THÊM DÒNG NÀY: Áp dụng sửa lỗi trước khi chấm điểm

# --- LOAD MODEL ---
print("⏳ Đang load model...")
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'ocr_plate.keras')

if not os.path.exists(model_path):
    print("❌ Không tìm thấy model!")
    exit()

@keras.saving.register_keras_serializable()
class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs): super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred): return y_pred

full_model = models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer})
prediction_model = models.Model(inputs=full_model.inputs[0], outputs=full_model.get_layer("dense_out").output)
print("✅ Load model thành công!")

# --- HÀM DỰ ĐOÁN ---
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Greedy Search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        # Làm sạch [UNK]
        res = res.replace('[UNK]', '').replace('_', '').strip()
        output_text.append(res)
    return output_text

# --- THAY THẾ HÀM load_test_data CŨ BẰNG HÀM NÀY ---
# --- THAY THẾ HÀM load_test_data CŨ BẰNG HÀM NÀY ---
def load_test_data(root_folder, label_file):
    img_paths = []
    labels = []
    
    if not os.path.exists(label_file): 
        print(f"❌ Không tìm thấy file txt: {label_file}")
        return [], []
    
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            # --- ĐOẠN SỬA QUAN TRỌNG ---
            # Chỉ lấy phần TÊN FILE (bỏ qua ./test/ ở đằng trước)
            # Ví dụ: ./test/anh1.jpg -> anh1.jpg
            file_name = os.path.basename(parts[0]) 
            
            label = parts[1].strip().upper()
            
            # Ghép trực tiếp vào thư mục chứa ảnh
            full_path = os.path.join(root_folder, file_name)
            
            if os.path.exists(full_path):
                img_paths.append(full_path)
                labels.append(label)
            else:
                # In ra lỗi đầu tiên để dễ debug
                if len(img_paths) == 0:
                    print(f"⚠️ Thử tìm: {full_path} -> KHÔNG THẤY!")
                    
    return img_paths, labels

# --- CHẠY ĐÁNH GIÁ ---
def evaluate(test_root, test_label):
    print(f"\n📂 Đang đọc dữ liệu test từ: {test_label}")
    paths, labels = load_test_data(test_root, test_label)
    
    total = len(paths)
    if total == 0:
        print("❌ Không tìm thấy ảnh test nào!")
        return

    print(f"⚡ Bắt đầu chấm điểm trên {total} ảnh...")
    
    correct_count = 0
    error_cases = [] # Lưu lại các ca sai để soi

    # Duyệt từng ảnh (Có thể batch nhưng loop cho dễ debug)
    for i in range(total):
        # 1. Xử lý ảnh
        img = tf.io.read_file(paths[i])
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.expand_dims(img, axis=0)

        # 2. Dự đoán
        preds = prediction_model.predict(img, verbose=0)
        probs = tf.nn.softmax(preds)
        pred_text = decode_batch_predictions(probs)[0]
        pred_text_corrected = autocorrect_plate(pred_text)
        
        ground_truth = labels[i].replace('-', '').replace('.', '') # Làm sạch label gốc luôn cho công bằng

        # So sánh cái đã sửa
        if pred_text_corrected == ground_truth:
            correct_count += 1
        else:
            # In ra để xem tại sao sai (quan trọng để debug)
            error_cases.append((ground_truth, pred_text_corrected, paths[i]))
        ground_truth = labels[i]

        # 3. So sánh
        if pred_text == ground_truth:
            correct_count += 1
        else:
            error_cases.append((ground_truth, pred_text, paths[i]))
        
        # In tiến độ kiểu pro (mỗi 10 ảnh)
        if (i+1) % 100 == 0:
            print(f"   Processed {i+1}/{total}...")

    # --- KẾT QUẢ ---
    accuracy = (correct_count / total) * 100
    print("\n" + "="*40)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ")
    print("="*40)
    print(f"✅ Tổng số ảnh: {total}")
    print(f"🎯 Số câu đúng: {correct_count}")
    print(f"❌ Số câu sai:  {total - correct_count}")
    print(f"⭐ ĐỘ CHÍNH XÁC (ACCURACY): {accuracy:.2f}%")
    print("="*40)

    if len(error_cases) > 0:
        print("\n💀 DANH SÁCH CÁC CÂU SAI (Top 20):")
        print(f"{'THỰC TẾ':<15} | {'MODEL ĐOÁN':<15} | {'FILE ẢNH'}")
        print("-" * 60)
        for gt, pred, path in error_cases[:20]: # Chỉ in 20 cái đầu
            fname = os.path.basename(path)
            print(f"{gt:<15} | {pred:<15} | {fname}")
        
        # Gợi ý fix lỗi
        print("\n💡 Gợi ý:")
        print("- Nếu sai ký tự giống nhau (8-B, 0-O): Cần dùng hàm autocorrect.")
        print("- Nếu sai hoàn toàn: Ảnh quá mờ hoặc nhiễu.")

if __name__ == "__main__":
    # 1. Trỏ thẳng vào folder chứa ảnh
    test_data_folder = r"E:\Project_OCR\data\test_a" 
    
    # 2. Trỏ vào file txt
    test_label_file = r"E:\Project_OCR\data\rec_gt_test.txt" 

    evaluate(test_data_folder, test_label_file)