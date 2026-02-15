from roboflow import Roboflow
import os
from dotenv import load_dotenv
load_dotenv()


rf = Roboflow(api_key= os.getenv("roboflow_api_recognize"))
project = rf.workspace("license-plate-reg").project("viet-nam-ocr-plate")
version = project.version(1)
dataset = version.download("yolo26")