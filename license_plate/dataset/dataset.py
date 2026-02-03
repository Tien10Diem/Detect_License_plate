from roboflow import Roboflow
rf = Roboflow(api_key="oDbZbKfbEeLG40fFsbDU")
project = rf.workspace("tran-ngoc-xuan-tin-k15-hcm-dpuid").project("vietnam-license-plate-h8t3n")
version = project.version(1)
dataset = version.download("yolo26")
                