import os
import csv
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, callbacks

# --- 1. CẤU HÌNH ---
IMG_WIDTH = 200
IMG_HEIGHT = 50
MAX_LABEL_LEN = 12
CHAR_LIST = "0123456789ABCDEFGHKLMNPSTUVXYZ-. JQORĐI" 

char_to_num = layers.StringLookup(vocabulary=list(CHAR_LIST), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
valid_chars = set(CHAR_LIST)

# --- 2. LOAD DATA ---
# ... (Phần import và cấu hình giữ nguyên) ...

# --- 2. LOAD DATA (CẬP NHẬT MỚI) ---

# Hàm đọc CSV cũ (Giữ nguyên)
def load_data_csv(img_folder, label_file):
    img_paths = []
    labels = []
    if not os.path.exists(label_file): return [], []
    with open(label_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2 or row[0] == "Name": continue
            filename = row[0].strip()
            label = row[1].strip().upper()
            if any(c not in valid_chars for c in label): continue
            path = os.path.join(img_folder, filename)
            if os.path.exists(path):
                img_paths.append(path)
                labels.append(label)
    return img_paths, labels

# 🔥 HÀM MỚI: Đọc file TXT (Data thật)
def load_data_txt(root_folder, label_file):
    img_paths = []
    labels = []
    if not os.path.exists(label_file): 
        print(f"⚠️ Không tìm thấy file label: {label_file}")
        return [], []
    
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split() # Tự động tách theo khoảng trắng hoặc Tab
            if len(parts) < 2: continue
            
            # Xử lý đường dẫn: ./train/abc.jpg -> train/abc.jpg
            rel_path = parts[0].replace('./', '').replace('/', os.sep)
            label = parts[1].strip().upper()
            
            # Ghép đường dẫn đầy đủ
            full_path = os.path.join(root_folder, rel_path)
            
            # Lọc ký tự lạ (nếu có)
            if any(c not in valid_chars for c in label): continue
            
            if os.path.exists(full_path):
                img_paths.append(full_path)
                labels.append(label)
            else:
                # Debug nhẹ nếu không thấy ảnh
                # print(f"⚠️ Thiếu ảnh: {full_path}") 
                pass
                
    return img_paths, labels

current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_dir, '..', 'data')

# 1. Load Data Cũ (Generated - Để model không quên bài cũ)
print("⏳ Đang load data cũ...")
paths_old, labels_old = load_data_csv(os.path.join(data_root, 'generated'), os.path.join(data_root, 'label_generated.csv'))

# --- TÌM VÀ SỬA ĐOẠN NÀY TRONG FILE TRAIN.PY ---

# 2. Load Data Mới (Real - QUAN TRỌNG)
# Thư mục gốc (Nơi chứa file txt và folder train)
new_data_folder = r"E:\Project_OCR\data" 

# Đường dẫn chính xác tới file label
new_label_file = r"E:\Project_OCR\data\rec_gt_train.txt"

print(f"⏳ Đang load data MỚI từ: {new_label_file}")
paths_new, labels_new = load_data_txt(new_data_folder, new_label_file)

# 3. Trộn lại (Ưu tiên data mới bằng cách nhân bản nó lên nếu nó ít)
# Mẹo: Nếu data thật ít (< 1000 ảnh), ta nhân đôi nó lên để Model học kỹ hơn
if len(paths_new) > 0:
    print(f"🔥 Tìm thấy {len(paths_new)} ảnh THẬT! (Nhân đôi trọng số)")
    paths_new = paths_new * 2 # Nhân đôi
    labels_new = labels_new * 2

img_paths = paths_old + paths_new
labels = labels_old + labels_new

# Shuffle kỹ
combined = list(zip(img_paths, labels))
import random
random.shuffle(combined)
img_paths, labels = zip(*combined)
img_paths, labels = list(img_paths), list(labels)

print(f"✅ TỔNG CỘNG: {len(img_paths)} ảnh (Cũ + Mới).")

if len(img_paths) == 0: exit()

# ... (Các phần sau giữ nguyên) ...

# --- 3. PREPROCESSING ---
def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # Tắt Augmentation để học ổn định trong giai đoạn cuối
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    pad_len = MAX_LABEL_LEN - tf.shape(label)[0]
    label = tf.pad(label, [[0, pad_len]], constant_values=99)
    return {"image": img, "label": label}

BATCH_SIZE = 32
full_dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
full_dataset = full_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)

split_idx = int(len(img_paths) * 0.9)
train_dataset = full_dataset.take(split_idx).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = full_dataset.skip(split_idx).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 4. MODEL & CALLBACKS ---
@keras.saving.register_keras_serializable()
class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones(shape=(batch_len,), dtype="int64")
        label_length = tf.cast(tf.math.count_nonzero(tf.not_equal(y_true, 99), axis=1), dtype="int64")
        loss = tf.nn.ctc_loss(tf.cast(y_true, "int32"), y_pred, label_length, tf.cast(input_length, "int32"), logits_time_major=False, blank_index=-1)
        self.add_loss(tf.reduce_mean(loss))
        return y_pred

class MonitorCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n🔎 Kết quả test Epoch {epoch+1}:")
        for batch in validation_dataset.take(1):
            imgs = batch['image']
            pred_model = models.Model(self.model.inputs[0], self.model.get_layer("dense_out").output)
            preds = pred_model.predict(imgs, verbose=0)
            input_len = np.ones(preds.shape[0]) * preds.shape[1]
            results = tf.keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
            for i in range(min(3, len(results))):
                res_str = tf.strings.reduce_join(num_to_char(results[i])).numpy().decode("utf-8")
                print(f"   🚗 Biển {i+1}: '{res_str}'")
            break

# --- 5. EXECUTION ---
print("🔄 Đang load model để TINH CHỈNH (Fine-tune)...")
save_path = os.path.join(current_dir, '..', 'model', 'ocr_plate.keras')

if os.path.exists(save_path):
    model = models.load_model(save_path, custom_objects={'CTCLayer': CTCLayer})
    
    # 🛑 QUAN TRỌNG: Chỉnh LR cực nhỏ (0.00005) để học chi tiết
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005)) 
    print("✅ Chế độ: Precision Mode (LR=0.00005)")
else:
    print("❌ Chưa có model cũ! Hãy chạy train từ đầu trước.")
    exit()

# Callback: Chỉ giảm LR khi thực sự cần thiết
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)
checkpoint = callbacks.ModelCheckpoint(save_path, monitor='loss', save_best_only=True, verbose=1)

print("🚀 BẮT ĐẦU FINE-TUNING (Mài cho sắc nét)...")
model.fit(train_dataset, validation_data=validation_dataset, epochs=20, callbacks=[MonitorCallback(), lr_scheduler, checkpoint])