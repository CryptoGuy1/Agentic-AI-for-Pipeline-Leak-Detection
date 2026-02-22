from ultralytics import YOLO

class VisionTool:

    def __init__(self, model_path):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)

    def detect(self, image_path):
        results = self.model(image_path)

        # get predicted class
        class_id = int(results[0].probs.top1)
        confidence = float(results[0].probs.top1conf)

        return class_id, confidence