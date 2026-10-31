
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "runs" / "detect" / "surfacescope_imgsz512"
    / "weights" / "best.pt"
)
IMAGE_DIR = ROOT / "data" / "processed" / "neu_yolo" / "images" / "validation"
LABEL_DIR = ROOT / "data" / "processed" / "neu_yolo" / "labels" / "validation"
REPORT_DIR = ROOT / "reports"

CLASSES = {0: "crazing", 4: "rolled-in_scale"}
CONFIDENCE = 0.25
IOU_MATCH_THRESHOLD = 0.50


def read_ground_truth(label_path):
    boxes = defaultdict(list)

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        cls, xc, yc, w, h = map(float, parts)
        cls = int(cls)

        if cls not in CLASSES:
            continue

        boxes[cls].append([
            xc - w / 2, yc - h / 2,
            xc + w / 2, yc + h / 2
        ])

    return boxes


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes):
    """Greedy, confidence-ordered, one-to-one IoU matching."""
    pred_boxes = sorted(pred_boxes, key=lambda item: item["confidence"], reverse=True)
    used_gt = set()
    tp = 0
    matched_ious = []
    unmatched_predictions = 0

    for prediction in pred_boxes:
        best_index = None
        best_iou = 0.0

        for index, gt_box in enumerate(gt_boxes):
            if index in used_gt:
                continue

            score = iou(prediction["box"], gt_box)
            if score > best_iou:
                best_iou = score
                best_index = index

        if best_index is not None and best_iou >= IOU_MATCH_THRESHOLD:
            tp += 1
            used_gt.add(best_index)
            matched_ious.append(best_iou)
        else:
            unmatched_predictions += 1

    fn = len(gt_boxes) - len(used_gt)
    fp = unmatched_predictions

    return tp, fp, fn, matched_ious


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    totals = {
        class_id: {"tp": 0, "fp": 0, "fn": 0, "ious": []}
        for class_id in CLASSES
    }

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    for image_path in image_paths:
        gt_by_class = read_ground_truth(
            LABEL_DIR / f"{image_path.stem}.txt"
        )

        result = model.predict(
            source=str(image_path),
            imgsz=512,
            conf=CONFIDENCE,
            device=0,
            verbose=False,
        )[0]

        pred_by_class = defaultdict(list)

        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in CLASSES:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            image_height, image_width = result.orig_shape

            pred_by_class[class_id].append({
                "box": [
                    xyxy[0] / image_width,
                    xyxy[1] / image_height,
                    xyxy[2] / image_width,
                    xyxy[3] / image_height,
                ],
                "confidence": float(box.conf.item()),
            })

        for class_id in CLASSES:
            tp, fp, fn, matched_ious = match_boxes(
                gt_by_class[class_id],
                pred_by_class[class_id],
            )

            totals[class_id]["tp"] += tp
            totals[class_id]["fp"] += fp
            totals[class_id]["fn"] += fn
            totals[class_id]["ious"].extend(matched_ious)

    rows = []

    for class_id, class_name in CLASSES.items():
        item = totals[class_id]
        tp, fp, fn = item["tp"], item["fp"], item["fn"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        rows.append({
            "class": class_name,
            "confidence_threshold": CONFIDENCE,
            "iou_threshold": IOU_MATCH_THRESHOLD,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "mean_matched_IoU": (
                float(np.mean(item["ious"])) if item["ious"] else 0.0
            ),
        })

    output = REPORT_DIR / "difficult_class_error_analysis_conf025.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    print("\nDIFFICULT-CLASS ERROR ANALYSIS")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved report: {output}")
    print(
        "\nNote: these are class-specific metrics at confidence 0.10 "
        "and IoU 0.50, not official COCO mAP."
    )


if __name__ == "__main__":
    main()
