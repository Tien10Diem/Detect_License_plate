import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models

# Tắt log rác
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class LicensePlateOCR:
    def __init__(self, model_path=None):
        # 1. Cấu hình
        self.img_width = 200
        self.img_height = 50
        self.char_list = "0123456789ABCDEFGHKLMNPSTUVXYZ-. JQORĐI"
        self.char_to_num = layers.StringLookup(vocabulary=list(self.char_list), mask_token=None)
        self.num_to_char = layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)
        
        # 2. Load Model
        if model_path is None:
            # Tự động tìm model trong folder dự án
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'model', 'ocr_plate.keras')
            
        print(f"⏳ Đang load OCR Model từ: {model_path}...")
        
        # Đăng ký lớp tùy chỉnh CTCLayer để không bị lỗi khi load
        @keras.saving.register_keras_serializable()
        class CTCLayer(layers.Layer):
            def __init__(self, name=None, **kwargs): super().__init__(name=name, **kwargs)
            def call(self, y_true, y_pred): return y_pred

        try:
            full_model = models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer})
            # Chỉ lấy phần dự đoán (bỏ lớp Loss)
            self.prediction_model = models.Model(inputs=full_model.inputs[0], outputs=full_model.get_layer("dense_out").output)
            print("✅ OCR System sẵn sàng!")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            self.prediction_model = None

    def preprocess_image(self, img_path):
        # Xử lý ảnh đầu vào
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.expand_dims(img, axis=0)
        return img

    def autocorrect(self, text):
        # --- CẢNH SÁT CHÍNH TẢ (Logic sửa lỗi) ---
        text = text.replace('[UNK]', '').replace('_', '').replace('-', '').replace('.', '').strip()
        
        # Mapping lỗi thường gặp
        dict_char_to_num = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'J': '3', 'L': '1', 'S': '5', 'B': '8', 'Z': '7', 'A': '4', 'G': '6'}
        dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '7': 'T', '6': 'G'}

        chars = list(text)
        new_chars = []

        for i, c in enumerate(chars):
            # Quy tắc 1: 2 ký tự đầu là SỐ (Mã tỉnh 59, 29...)
            if i < 2:
                new_chars.append(dict_char_to_num.get(c, c))
            
            # Quy tắc 2: Ký tự thứ 3 là CHỮ (Series L, H, K...)
            elif i == 2:
                if c.isdigit():
                    new_chars.append(dict_num_to_char.get(c, c)) # Ép số thành chữ
                else:
                    new_chars.append(c)
            
            # Quy tắc 3: Các ký tự sau là SỐ
            elif i > 2:
                new_chars.append(dict_char_to_num.get(c, c))
                
        return "".join(new_chars)

    def predict(self, image_path):
        if self.prediction_model is None: return "Error: Model not loaded"
        
        if not os.path.exists(image_path): return "Error: Image not found"

        # 1. Nhìn (Model chạy)
        img_tensor = self.preprocess_image(image_path)
        preds = self.prediction_model.predict(img_tensor, verbose=0)
        probs = tf.nn.softmax(preds)
        
        # 2. Dịch (Decode CTC)
        input_len = np.ones(probs.shape[0]) * probs.shape[1]
        results = tf.keras.backend.ctc_decode(probs, input_length=input_len, greedy=True)[0][0]
        res_str = tf.strings.reduce_join(self.num_to_char(results[0])).numpy().decode("utf-8")
        
        # 3. Sửa (Autocorrect)
        final_text = self.autocorrect(res_str)
        
        return final_text

# --- TEST CHẠY THỬ (Nếu chạy trực tiếp file này) ---
if __name__ == "__main__":
    # Khởi tạo bộ máy
    ocr = LicensePlateOCR()
    
    # Test thử 1 ảnh
    test_img = r"C:\Users\ASUS\Hoc_DL\learning-DL\Detect_Lisence_plate\OCR\data\cropped\car_14.jpg" # Thay ảnh của bạn vào
    
    result = ocr.predict(test_img)
    print("="*30)
    print(f"🚗 KẾT QUẢ CUỐI CÙNG: {result}")
    print("="*30)