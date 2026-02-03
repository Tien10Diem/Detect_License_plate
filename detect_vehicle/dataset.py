from roboflow import Roboflow

if __name__ == "__main__":
    
    rf = Roboflow(api_key="oDbZbKfbEeLG40fFsbDU")
    project = rf.workspace("haha-hxiwc").project("vehicle-j080n-wpc58")
    version = project.version(1)
    dataset = version.download("yolo26")