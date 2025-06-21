
from ultralytics import YOLO

class YoloDetector: #a class defined to tie to YOLOv11 model best.pt 
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["ball", "goalkeeper", "player", "referee"] #the detected objects from the best.pt model's definition
        self.confidence = confidence

    def track(self, image, tracker_config="botsort.yaml"):# object tracking using botsort to ensure IDs dont change for each frame of the video
        results = self.model.track(
            source=image,
            conf=self.confidence,
            persist=True,
            tracker=tracker_config,
            stream=True  # allows us to use it on single frames
        )
        return results
