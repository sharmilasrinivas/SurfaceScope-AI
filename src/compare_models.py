from pathlib import Path
import csv
import time

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "neu_yolo" / "data.yaml"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

MODELS = [
    {
        "name": "YOLOv8n_baseline_320_ep30",
        "weights": ROOT / "runs/detect/surfacescope_baseline/weights/best.pt",
        "imgsz": 320,
        "epochs": 30,
    },
    {
        "name": "YOLOv8n_512_ep30",
        "weights": ROOT / "runs/detect/surfacescope_imgsz512/weights/best.pt",
        "imgsz": 512,
        "epochs": 30,
    },
    {
        "name": "YOLOv8n_512_ep50",
        "weights": ROOT / "runs/detect/surfacescope_imgsz512-2/weights/best.pt",
        "imgsz": 512,
        "epochs": 50,
    },
]

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA}")

    rows = []

    for item in MODELS:
        weights = item["weights"]

        if not weights.exists():
            print(f"SKIPPED: {item['name']} — missing {weights}")
            continue

        print(f"\nEvaluating: {item['name']}")
        print(f"Weights: {weights}")
        print(f"Image size: {item['imgsz']}")

        model = YOLO(str(weights))

        start = time.perf_counter()
        metrics = model.val(
            data=str(DATA),
            imgsz=item["imgsz"],
            device=0,
            plots=False,
            verbose=False,
            project=str(ROOT / "runs" / "comparison"),
            name=item["name"],
            exist_ok=True,
        )
        elapsed = time.perf_counter() - start

        row = {
            "model": item["name"],
            "weights": str(weights.relative_to(ROOT)),
            "epochs": item["epochs"],
            "imgsz": item["imgsz"],
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
            "map50": round(float(metrics.box.map50), 6),
            "map50_95": round(float(metrics.box.map), 6),
            "validation_seconds": round(elapsed, 2),
        }
        rows.append(row)

        print(
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f} | "
            f"mAP50={row['map50']:.4f} | "
            f"mAP50-95={row['map50_95']:.4f} | "
            f"Time={elapsed:.1f}s"
        )

    if not rows:
        raise RuntimeError("No models were evaluated. Check the weight paths.")

    output = REPORTS / "model_comparison.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 75)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 75)

    for row in rows:
        print(
            f"{row['model']:28} "
            f"mAP50={row['map50']:.4f} "
            f"mAP50-95={row['map50_95']:.4f}"
        )

    print(f"\nCSV saved to: {output}")

if __name__ == "__main__":
    main()
