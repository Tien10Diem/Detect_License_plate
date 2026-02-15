from roboflow import Roboflow
import os
from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    
    rf = Roboflow(api_key=os.getenv("roboflow_api_vehicle"))
    project = rf.workspace("ta-nuoog").project("yolov8-2-iao3v")
    version = project.version(1)
    dataset = version.download("yolo26")