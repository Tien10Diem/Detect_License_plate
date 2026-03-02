# License Plate Recognition (LPR) System with YOLO and Docker

This is a comprehensive Computer Vision project for detecting vehicles, locating license plates, and recognizing characters on license plates from a video source. The entire processing pipeline is designed for optimal performance and is fully containerized using Docker. The project includes two model versions: `base` and `improved` (enhanced with techniques like CBAM).

## Demo Video

[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://youtu.be/ZiOYWaxIXFE)

## Base vs. Improved Comparison

- base video:
![base](sample/output/output_base.mp4)

- improved vs. base license plate detection (conf=0.5 and iou=0.5):
![improved](sample/lp_imp.jpg)
![base](sample/lp_base.jpg)

- improved vs. base character recognition (conf=0.5 and iou=0.5):
![improved](sample/imp_rec.jpg)
![base](sample/base_rec.jpg)


## Key Technical Features

- **Vehicle Detection & Tracking:** Uses a YOLO model combined with the ByteTrack algorithm to detect and maintain a unique ID for each vehicle throughout the video.
- **License Plate Detection:** Accurately crops the license plate region based on the vehicle's bounding box coordinates.
- **Character Recognition (OCR) & Spatial Arrangement:** Recognizes characters and applies a dynamic line grouping algorithm to handle slanted license plates.
- **Performance Optimization:** The OCR model is executed only once every 5 frames to reduce computational load.
- **Data Interpolation:** Utilizes Linear Interpolation via Pandas to fill in missing frames from tracking, ensuring a smooth trajectory.

## Project Structure

```text
.
├── data/                 # Data and configuration scripts for the models
├── docker-compose.yml    # Docker Compose configuration (including GPU setup)
├── Dockerfile            # Docker environment blueprint
├── requirements.txt      # Required Python libraries
├── sample/               # Contains input videos (raw_video/) and output videos (output/)
├── scripts/              # Scripts for model training
│   ├── base/             # Training scripts for the base model
│   └── improved/         # Training scripts for the improved model
├── src/                  # Core source code
│   ├── models/           # Model architecture definitions, configs, and custom blocks (CBAM)
│   ├── pipeline/         # Main processing logic (inference.py, inference_teacher.py)
│   └── utils/            # Utility functions (e.g., convertOnnx.py)
└── weights/              # Trained model weights
    ├── base/             # Weights for the base models
    ├── model_license_plate/ # Weights for the license plate detection model
    ├── model_numbers/    # Weights for the character recognition model
    └── model_vehicle/    # Weights for the vehicle detection model
```

## Getting Started

### Local Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    The main libraries include `ultralytics`, `torch`, `opencv-python`, `pandas`, and `onnxruntime-gpu`.

## Usage (Inference)

### Method 1: Run via Docker
This method automatically sets up the environment and GPU configuration. Place your video in `sample/raw_video/` and run:

```bash
docker-compose up --build
```
The resulting video will be saved in the `sample/output/` directory.

### Method 2: Run directly with Python
Use the `src/pipeline/inference.py` script with parameters for customization. The following command uses the default ONNX weights to process a video:

```bash
python src/pipeline/inference.py --video_path sample/raw_video/video_cut.mp4 --output_path sample/output/result.mp4 --gap 150
```

## Utilities

### Cut Video

The project provides the `cutvideo.py` script to cut a segment from a source video file, which is useful for quick testing.

## Training

Scripts for training the individual models are located in the `scripts/` directory. You can choose to train the `base` or `improved` version based on your needs.

-   `scripts/base/base_vehicle.py`: Train the vehicle detection model (base version).
-   `scripts/improved/train_vehicle.py`: Train the vehicle detection model (improved version).
-   The same applies to the `..._lp.py` (license plate detection) and `..._reNum.py` (character recognition) scripts.

## Models

The project uses three separate YOLO models for each task:
1.  **Vehicle Detection:** `vehicle_detector` model.
2.  **License Plate Detection:** `plate_detector` model.
3.  **Character Recognition:** `recognize_number` model.

Each model has a `base` and an `improved` version. The `improved` version may use custom architectures like `YOLO26m` with `CBAM` blocks (defined in `src/models/`). For optimal inference speed, it is recommended to use the versions that have been converted to the **ONNX** format.

### Some Notes:
- The character recognition model performs better on car license plates compared to motorcycle plates.
- The interpolation process can be slow when there are many objects in the frame.
