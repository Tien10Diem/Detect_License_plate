# Hệ thống Nhận diện và Theo dõi Biển số xe (LPR) với YOLO26 và Docker

Đây là dự án toàn diện ứng dụng Computer Vision để phát hiện phương tiện, khoanh vùng biển số và nhận dạng các ký tự trên biển số từ video. Toàn bộ luồng xử lý (pipeline) được thiết kế tối ưu hiệu năng và đóng gói hoàn chỉnh bằng Docker.

![Video Demo](sample/output/output.mp4)
## Tính năng kỹ thuật nổi bật

- **Phát hiện & Theo dõi xe (Vehicle Detection & Tracking):** Ứng dụng mô hình YOLO26m kết hợp thuật toán ByteTrack để phát hiện và duy trì ID độc nhất cho từng phương tiện xuyên suốt video.
- **Phát hiện biển số xe (License Plate Detection):** Trích xuất (crop) chính xác vùng chứa biển số dựa trên tọa độ bounding box của phương tiện.
- **Nhận dạng ký tự (OCR) & Sắp xếp không gian:** - Nhận diện ký tự và áp dụng thuật toán gom dòng động (chống lỗi biển số nghiêng).
  - Tối ưu hóa hiệu năng: Chỉ thực thi model OCR mỗi 5 frame để giảm tải tính toán.
- **Nội suy dữ liệu (Data Interpolation):** Ứng dụng thuật toán nội suy tuyến tính (Linear Interpolation) qua Pandas để lấp đầy các khung hình mất dấu, đảm bảo quỹ đạo mượt mà. Đã tích hợp kỹ thuật kiểm soát kích thước cửa sổ trượt để tránh ngoại lệ tương thích dữ liệu.

## Cấu trúc dự án

```text
.
├── data/                 # Scripts và dữ liệu cấu hình cho phát hiện/nhận dạng
├── docker-compose.yml    # Cấu hình Docker Compose (bao gồm thiết lập GPU)
├── Dockerfile            # Bản thiết kế môi trường cho ứng dụng
├── requirements.txt      # Các thư viện Python cần thiết (ultralytics, pandas, onnxruntime-gpu...)
├── sample/               # Chứa video đầu vào (raw_video/) và video đầu ra (output/)
├── scripts/              # Các script phục vụ huấn luyện mô hình
├── src/                  # Mã nguồn lõi
│   ├── models/           # Định nghĩa cấu trúc và các khối tùy chỉnh của mô hình
│   ├── pipeline/         # Logic luồng xử lý chính (inference.py)
│   └── utils/            # Các hàm tiện ích hỗ trợ
└── weights/              # Các trọng số model đã huấn luyện (.onnx)
```

## Bắt đầu

### Cài đặt môi trường Local

1.  **Clone repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

## Sử dụng (Inference)

### Cách 1: Chạy qua Docker 
Phương pháp này tự động thiết lập môi trường và cấu hình GPU mà không cần cài đặt phức tạp trên máy host. Đặt video vào `sample/raw_video/` và chạy:

```bash
docker-compose up --build
```
Video kết quả sẽ tự động được xuất ra tại thư mục `sample/output/`.

### Cách 2: Chạy trực tiếp bằng Python
Sử dụng script `inference.py` kèm các tham số dòng lệnh (`argparse`) để linh hoạt điều chỉnh luồng đầu vào/đầu ra:

```bash
python src/pipeline/inference.py --video_path sample/raw_video/video_cut.mp4 --output_path sample/output/result.mp4 --gap 150
```

## Huấn luyện (Training)

Các script để huấn luyện (train) các model riêng biệt được đặt trong thư mục `scripts/`.

-   `scripts/train_vehicle.py`: Huấn luyện mô hình phát hiện xe.
-   `scripts/train_lp.py`: Huấn luyện mô hình phát hiện biển số.
-   `scripts/train_reNum.py`: Huấn luyện mô hình nhận dạng ký tự (OCR).

## Các Model

Dự án này sử dụng kiến trúc **YOLO26m** cho cả 3 tác vụ: phát hiện phương tiện, khoanh vùng biển số và nhận diện ký tự. Cấu hình model và các khối tùy chỉnh có thể được tìm thấy trong thư mục `src/models/`. Để tối ưu tốc độ Inference trên môi trường thực tế, các trọng số phân phối trong thư mục `weights/` đều được chuyển đổi và vận hành dưới định dạng **ONNX Runtime**.

### Một số lưu ý:
- Model nhận diện các ký tự trong biển số thật sự chưa ổn đối với xe máy nhưng khá tốt với ô tô.
- Việc nội suy (interpolated) khá chậm, đặc biệt là đối với các frame nhiều object.
