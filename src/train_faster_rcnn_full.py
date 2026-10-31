
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "data" / "processed" / "neu_yolo" / "data.yaml"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
NUM_CLASSES = len(CLASS_NAMES) + 1  # Background is class 0
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_paths():
    with DATA_YAML.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root = Path(config["path"])
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()

    paths = {}
    for split in ("train", "val"):
        image_dir = (root / config[split]).resolve()
        label_dir = root / "labels" / image_dir.name

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        paths[split] = (image_dir, label_dir)

    return paths


class SteelDefectDataset(Dataset):
    def __init__(self, image_dir, label_dir, limit=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        )

        if limit is not None:
            self.images = self.images[:limit]

        if not self.images:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = [], []

        if label_path.exists():
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid YOLO annotation at {label_path}:{line_number}"
                    )

                class_id, xc, yc, bw, bh = map(float, parts)
                class_id = int(class_id)

                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    raise ValueError(
                        f"Invalid class ID {class_id} in {label_path}"
                    )

                # YOLO normalized xywh -> pixel xyxy.
                x1 = (xc - bw / 2) * width
                y1 = (yc - bh / 2) * height
                x2 = (xc + bw / 2) * width
                y2 = (yc + bh / 2) * height

                x1 = max(0.0, min(float(width), x1))
                y1 = max(0.0, min(float(height), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(class_id + 1)  # TorchVision reserves 0 for background.

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor(
                [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        return to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=320,
        max_size=320,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )
    return model


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()

        if step == 1 or step % 100 == 0 or step == len(loader):
            print(
                f"Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss {loss.item():.4f}"
            )

    return running_loss / max(1, len(loader))


def validation_loss(model, loader, device):
    # TorchVision detection models return losses in training mode.
    # Save and restore module training flags so evaluation doesn't leave
    # the model in training mode.
    training_states = [(module, module.training) for module in model.modules()]
    model.train()
    total_loss = 0.0

    try:
        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) for k, v in target.items()}
                    for target in targets
                ]
                loss_dict = model(images, targets)
                total_loss += sum(loss_dict.values()).item()
    finally:
        for module, was_training in training_states:
            module.training = was_training

    return total_loss / max(1, len(loader))


def box_iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return torch.zeros(0)

    left_top = torch.maximum(box[:2], boxes[:, :2])
    right_bottom = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (
        (right_bottom - left_top).clamp(min=0).prod(dim=1)
    )

    area_a = (box[2:] - box[:2]).clamp(min=0).prod()
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    union = area_a + area_b - intersection

    return intersection / union.clamp(min=1e-9)


def evaluate_detector(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[x / 100 for x in range(50, 100, 5)],
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True,
    )

    # Keep predictions and targets for transparent fixed-threshold metrics.
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for images, targets in loader:
            images_device = [img.to(device) for img in images]
            outputs = model(images_device)

            for output, target in zip(outputs, targets):
                pred = {
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                }
                gt = {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                    "area": target["area"].cpu(),
                    "iscrowd": target["iscrowd"].cpu(),
                }

                metric.update([pred], [gt])
                all_predictions.append(pred)
                all_targets.append(gt)

    result = metric.compute()

    def scalar(value):
        return float(value.item()) if torch.is_tensor(value) else float(value)

    summary = {
        "map50": scalar(result["map_50"]),
        "map50_95": scalar(result["map"]),
        "mar100": scalar(result["mar_100"]),
    }

    # Fixed confidence threshold and IoU threshold of 0.50.
    class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES, start=1):
        tp = fp = fn = 0

        for pred, gt in zip(all_predictions, all_targets):
            pred_mask = (
                (pred["labels"] == class_id)
                & (pred["scores"] >= CONFIDENCE_THRESHOLD)
            )
            gt_mask = gt["labels"] == class_id

            pboxes = pred["boxes"][pred_mask]
            gboxes = gt["boxes"][gt_mask]

            if len(pboxes):
                order = pred["scores"][pred_mask].argsort(descending=True)
                pboxes = pboxes[order]

            matched_gt = set()
            for pbox in pboxes:
                overlaps = box_iou_one_to_many(pbox, gboxes)
                if len(overlaps):
                    for gi in overlaps.argsort(descending=True).tolist():
                        if (
                            overlaps[gi].item() >= IOU_THRESHOLD
                            and gi not in matched_gt
                        ):
                            matched_gt.add(gi)
                            tp += 1
                            break
                    else:
                        fp += 1
                else:
                    fp += 1

            fn += len(gboxes) - len(matched_gt)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        ap = float("nan")
        metric_classes = result.get("classes")
        metric_ap = result.get("map_per_class")
        if metric_classes is not None and metric_ap is not None:
            matches = (metric_classes == class_id).nonzero().flatten()
            if len(matches):
                ap = scalar(metric_ap[matches[0]])

        class_rows.append({
            "class": class_name,
            "ap50_95": ap,
            "precision_conf025_iou050": precision,
            "recall_conf025_iou050": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    return summary, class_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PyTorch:", torch.__version__)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    paths = get_dataset_paths()
    train_dir, train_labels = paths["train"]
    val_dir, val_labels = paths["val"]

    train_ds = SteelDefectDataset(train_dir, train_labels)
    val_ds = SteelDefectDataset(val_dir, val_labels)

    if args.smoke_test:
        train_ds = SteelDefectDataset(train_dir, train_labels, limit=4)
        val_ds = SteelDefectDataset(val_dir, val_labels, limit=4)

    print("Training images:", len(train_ds))
    print("Validation images:", len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )

    if args.smoke_test:
        output_dir = PROJECT_ROOT / "runs" / "faster_rcnn" / "smoke_test_metrics_v2"
    else:
        output_dir = PROJECT_ROOT / "runs" / "faster_rcnn" / "surfacescope_fasterrcnn_ep50"

    checkpoint_path = output_dir / "last_checkpoint.pth"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}\n"
            "Choose a new output directory or explicitly resume the existing run."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model().to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001, momentum=0.9, weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1
    )

    start_epoch = 1
    best_map = -1.0

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_map = checkpoint.get("best_map50_95", -1.0)
        print(f"Resuming from epoch {start_epoch}")

    epochs = 1 if args.smoke_test else args.epochs
    history_path = output_dir / "history.csv"
    class_metrics_path = output_dir / "class_metrics.csv"

    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss", "map50", "map50_95",
                "mar100", "seconds",
            ])

    if start_epoch > epochs:
        print("Checkpoint already completed the requested number of epochs.")
        return

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_loss = validation_loss(model, val_loader, device)
        summary, class_rows = evaluate_detector(model, val_loader, device)
        scheduler.step()
        duration = time.time() - start

        print(
            f"\nEpoch {epoch}/{epochs} | Train loss={train_loss:.4f} | "
            f"Val loss={val_loss:.4f} | mAP50={summary['map50']:.4f} | "
            f"mAP50-95={summary['map50_95']:.4f} | "
            f"mAR100={summary['mar100']:.4f} | "
            f"Time={duration / 60:.1f} min"
        )

        print("\nPer-class metrics (confidence >= 0.25, IoU >= 0.50):")
        for row in class_rows:
            print(
                f"{row['class']}: AP={row['ap50_95']:.4f}, "
                f"P={row['precision_conf025_iou050']:.3f}, "
                f"R={row['recall_conf025_iou050']:.3f}"
            )

        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss, summary["map50"],
                summary["map50_95"], summary["mar100"], duration,
            ])

        with class_metrics_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
            writer.writeheader()
            writer.writerows(class_rows)

        is_best = summary["map50_95"] > best_map
        if is_best:
            best_map = summary["map50_95"]
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print("Saved new best model by validation mAP@50-95.")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map50_95": best_map,
            "class_names": CLASS_NAMES,
        }, checkpoint_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nRun finished.")
    print("Output directory:", output_dir)
    print("Training history:", history_path)
    print("Per-class metrics:", class_metrics_path)
    print("Best model:", output_dir / "best_model.pth")
    print("Latest checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
