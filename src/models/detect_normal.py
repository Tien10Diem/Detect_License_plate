from ultralytics import YOLO

class DetectNormal:
    def __init__(self, weights_path, conf_thres=0.5):
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres

    def predict(self, image_tensor):
        return self.model.predict(source=image_tensor, conf=self.conf_thres, verbose=False)
        
    