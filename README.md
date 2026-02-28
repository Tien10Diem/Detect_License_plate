# License Plate Recognition (LPR) and Tracking System with YOLO26 and Docker

This is a comprehensive project applying Computer Vision to detect vehicles, localize license plates, and recognize characters on license plates from videos. The entire processing pipeline is performance-optimized and fully containerized using Docker.

![Video Demo](sample/output/output.mp4)
## Key Technical Features

- **Vehicle Detection & Tracking:** Utilizes the YOLO26m model combined with the ByteTrack algorithm to detect and maintain unique IDs for each vehicle throughout the video.
- **License Plate Detection:** Accurately extracts (crops) the license plate region based on the vehicle's bounding box coordinates.
- **Character Recognition (OCR) & Spatial Sorting:** - Recognizes characters and applies a dynamic line grouping algorithm (handling tilted license plates).
  - Performance optimization: Only executes the OCR model every 5 frames to reduce computational load.
- **Data Interpolation:** Applies Linear Interpolation via Pandas to fill in missing frames (missed detections), ensuring smooth trajectories. Integrated a sliding window size control technique to prevent data compatibility exceptions.

## Project Structure

```text
.
├── data/                 # Scripts and configuration data for detection/recognition
├── docker-compose.yml    # Docker Compose configuration (including GPU setup)
├── Dockerfile            # Environment blueprint for the application
├── requirements.txt      # Required Python libraries (ultralytics, pandas, onnxruntime-gpu...)
├── sample/               # Contains input videos (raw_video/) and output videos (output/)
├── scripts/              # Scripts for model training
├── src/                  # Core source code
│   ├── models/           # Model architecture definitions and custom blocks
│   ├── pipeline/         # Main processing pipeline logic (inference.py)
│   └── utils/            # Utility and support functions
└── weights/              # Trained model weights (.onnx)
```

## Getting Started

### Local Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage (Inference)

### Method 1: Running via Docker
This method automatically sets up the environment and GPU configuration without complex installations on the host machine. Place the video in `sample/raw_video/` and run:

```bash
docker-compose up --build
```
The resulting video will be automatically exported to the `sample/output/` directory.

### Method 2: Running directly with Python
Use the `inference.py` script with command-line arguments (`argparse`) to flexibly adjust the input/output pipeline:

```bash
python src/pipeline/inference.py --video_path sample/raw_video/video_cut.mp4 --output_path sample/output/result.mp4 --gap 150
```

## Training

Scripts to train individual models are located in the `scripts/` directory.

-   `scripts/train_vehicle.py`: Train the vehicle detection model.
-   `scripts/train_lp.py`: Train the license plate detection model.
-   `scripts/train_reNum.py`: Train the character recognition (OCR) model.

## Models

This project uses the **YOLO26m** architecture for all 3 tasks: vehicle detection, license plate localization, and character recognition. Model configurations and custom blocks can be found in the `src/models/` directory. To optimize inference speed in a production environment, the distributed weights in the `weights/` directory have been converted to and operate under the **ONNX Runtime** format.

### Important Notes:
- The character recognition model is currently sub-optimal for motorcycles but performs quite well on cars.
- The data interpolation process is rather slow, especially for frames containing a large number of objects.