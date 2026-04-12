from pathlib import Path
from ultralytics import YOLO

# Canonical gas mapping used everywhere else in the project
GAS_MAP = {
    "NoGas": 0,
    "Smoke": 1,
    "Mixture": 2,
    "Perfume": 3,
}

# IMPORTANT:
# YOLO internal class order is NOT the same as GAS_MAP.
# This mapping must match your trained YOLO classifier.
YOLO_IDX_TO_LABEL = {
    0: "Mixture",
    1: "NoGas",
    2: "Perfume",
    3: "Smoke",
}

YOLO_IDX_TO_GAS_ID = {
    idx: GAS_MAP[label] for idx, label in YOLO_IDX_TO_LABEL.items()
}


class VisionTool:
    """
    YOLOv8 classification wrapper for thermal gas images.

    Returns a dictionary with:
    - gas_id: canonical gas id used by the rest of the system
    - gas_name: canonical gas label
    - yolo_class_idx: raw YOLO class index
    - yolo_class_label: YOLO class label
    - confidence: top-1 confidence
    - image_path: source image path
    """

    def __init__(self, model_path: str):
        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(model_path)

    def predict(self, image_path: str) -> dict:
        image_path = str(image_path)

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(image_path, verbose=False)

        if not results:
            raise ValueError(f"No YOLO results returned for image: {image_path}")

        r = results[0]
        probs = getattr(r, "probs", None)

        # Safe fallback when classifier probabilities are missing
        if probs is None:
            return {
                "gas_id": GAS_MAP["NoGas"],
                "gas_name": "NoGas",
                "yolo_class_idx": 1,
                "yolo_class_label": "NoGas",
                "confidence": 0.0,
                "image_path": image_path,
                "note": "No classification probabilities returned; defaulted to NoGas",
            }

        yolo_class_idx = int(probs.top1)
        confidence = float(probs.top1conf)

        if yolo_class_idx not in YOLO_IDX_TO_LABEL:
            raise ValueError(
                f"Unexpected YOLO class index: {yolo_class_idx}. "
                f"Expected one of {list(YOLO_IDX_TO_LABEL.keys())}."
            )

        gas_name = YOLO_IDX_TO_LABEL[yolo_class_idx]
        gas_id = YOLO_IDX_TO_GAS_ID[yolo_class_idx]

        return {
            "gas_id": gas_id,
            "gas_name": gas_name,
            "yolo_class_idx": yolo_class_idx,
            "yolo_class_label": gas_name,
            "confidence": confidence,
            "image_path": image_path,
            "note": "YOLO classification prediction",
        }
