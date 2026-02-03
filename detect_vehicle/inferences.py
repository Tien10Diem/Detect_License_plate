import cv2
from ultralytics import YOLO
import argparse


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default= r'detect_vehicle\transfer_vehicle\best.pt', type=str, help='model path')
    parser.add_argument('--video', '-v', default= r'Video xe máy - xe hơi chạy trên đường - YouTube.mp4', type=str, help='video path')
    parser.add_argument('--conf', '-c', default= 0.25, type=float, help='confidence')

    return parser.parse_args()




class inf_video():
  def __init__(self, path_video, best_model, conf = 0.25):
    super().__init__()
    self.path_video = path_video
    self.model = YOLO(best_model)
    self.conf = conf

  def predict(self, frame):
    results = self.model(frame, imgsz=640, conf = self.conf)
    result_img = results[0].plot()
    return result_img

  def read_video(self):
    cap = cv2.VideoCapture(self.path_video)

    if not cap.isOpened():
      print("Error opening video file")
      return

    while True:
      flag, frame = cap.read()

      if not flag:
        break

      yield frame

    cap.release()

  def run(self):

    video = self.read_video()

    for frame in video:

      output_img = self.predict(frame)

      cv2.imshow("output",output_img)
      cv2.waitKey(1)


if __name__ == "__main__":
    args = args()
    model = args.model
    path_video = args.video
    conf = args.conf

    inf = inf_video(path_video, model, conf)

    inf.run()

