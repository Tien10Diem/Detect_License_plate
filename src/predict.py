import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models

# Tắt cảnh báo oneDNN và TF deprecated
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# --- 1. CẤU HÌNH (PHẢI KHỚP VỚI FILE TRAIN) ---
IMG_WIDTH = 224   
IMG_HEIGHT = 64
CHAR_LIST = "0123456789ABCDEFGHKLMNPSTUVXYZ-. "

char_to_num = layers.StringLookup(vocabulary=list(CHAR_LIST), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# --- 2. ĐỊNH NGHĨA LẠI CUSTOM LAYER ---
@keras.saving.register_keras_serializable()
class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # Không cần loss_fn khi predict

    def call(self, y_true, y_pred):
        return y_pred

# --- 3. LOAD MODEL ---
print("⏳ Đang load model...")
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'ocr_plate.keras')

if not os.path.exists(model_path):
    print(f"❌ Lỗi: Không tìm thấy file model tại {model_path}")
    exit()

try:
    # Load toàn bộ model
    full_model = models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer})
    
    # --- ĐOẠN SỬA LỖI QUAN TRỌNG ---
    # Thay vì lấy input từ layer, ta lấy input đầu tiên của model tổng (full_model.inputs[0])
    # Model lúc train có 2 inputs: [0]=Image, [1]=Label. Ta chỉ cần cái [0].
    prediction_model = models.Model(
        inputs=full_model.inputs[0], 
        outputs=full_model.get_layer(name="dense_out").output
    )
    print("✅ Load model thành công!")
    
except Exception as e:
    print(f"❌ Lỗi load model chi tiết: {e}")
    exit()

# --- 4. HÀM DỰ ĐOÁN ---
def predict_image(image_path):
    if not os.path.exists(image_path):
        return "File không tồn tại"
    
    # --- Xử lý ảnh ---
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    
    # DEBUG: Lưu ảnh mà model thực sự nhìn thấy để kiểm tra
    # Bạn mở file 'debug_input.jpg' lên xem nó có bị méo hay đen thui không
    debug_img = tf.cast(img * 255, tf.uint8).numpy()
    cv2.imwrite("debug_input.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    print("📸 Đã lưu ảnh đầu vào model tại: debug_input.jpg (Hãy mở lên xem!)")

    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)

    # --- Dự đoán ---
    preds = prediction_model.predict(img, verbose=0)
    
    # In ra xác suất thô (Raw Logits) để xem model có tự tin không
    # Lấy bước thời gian đầu tiên, in ra top 3 ký tự có xác suất cao nhất
    first_step_probs = tf.nn.softmax(preds[0][0])
    top_values, top_indices = tf.math.top_k(first_step_probs, k=3)
    print(f"📊 Tại bước 1, Model dự đoán index: {top_indices.numpy()} với độ tin cậy {top_values.numpy()}")
    
    # Giải mã
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    results = tf.keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
    
    # Convert sang text
    output_text = []
    for res in results:
        # Index 0 thường là [UNK] nếu config StringLookup mặc định
        # Index -1 là Blank
        print(f"🔢 Chuỗi Index giải mã được: {res.numpy()}") 
        res_str = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res_str)
        
    return output_text[0]

# --- 5. CHẠY THỬ ---
if __name__ == "__main__":
    # Thay đường dẫn ảnh của bạn (Dùng r"..." để tránh lỗi)
    test_img = r"E:\Project_OCR\data\test\bien-so-xe_0401085240.jpg"
    
    print(f"🔍 Đang nhận diện: {test_img}")
    
    if os.path.exists(test_img):
        try:
            result = predict_image(test_img)
            print("-------------------------------")
            print(f"🚗 BIỂN SỐ: {result}")
            print("-------------------------------")
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán: {e}")
    else:
        print("❌ Lỗi: Đường dẫn ảnh không đúng!")