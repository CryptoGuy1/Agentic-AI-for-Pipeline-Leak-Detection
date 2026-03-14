from ultralytics import YOLO

class VisionTool:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image):

        results = self.model(image)[0]

        # classification probabilities
        probs = results.probs

        if probs is None:
            return "nogas", 0.0

        class_id = int(probs.top1)
        confidence = float(probs.top1conf)

        label = self.model.names[class_id]

        return label, confidence