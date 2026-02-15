from roboflow import Roboflow
import os
from dotenv import load_dotenv
load_dotenv()

rf = Roboflow(api_key=os.getenv("roboflow_api_license_plate"))
project = rf.workspace("tran-ngoc-xuan-tin-k15-hcm-dpuid").project("vietnam-license-plate-h8t3n")
version = project.version(1)
dataset = version.download("yolo26")
                