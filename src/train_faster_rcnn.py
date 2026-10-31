
from pathlib import Path
import yaml
import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "processed" / "neu_yolo" / "data.yaml"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
NUM_CLASSES = len(CLASS_NAMES) + 1  # Background + 6 defects


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config missing: {DATA_YAML}")

    with DATA_YAML.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config.get("path", DATA_YAML.parent))
    if not dataset_root.is_absolute():
        dataset_root = (DATA_YAML.parent / dataset_root).resolve()

    train_images = dataset_root / config["train"]
    if not train_images.exists():
        raise FileNotFoundError(f"Training images missing: {train_images}")

    image_files = sorted(
        p for p in train_images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_files:
        raise RuntimeError(f"No training images found in {train_images}")

    image_path = image_files[0]
    label_path = (
        dataset_root / "labels" / "train" / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        raise FileNotFoundError(f"YOLO label file missing: {label_path}")

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    boxes = []
    labels = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label: {line}")

        class_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"Invalid class ID {class_id} in {label_path}")

        x1 = max(0.0, (xc - bw / 2) * width)
        y1 = max(0.0, (yc - bh / 2) * height)
        x2 = min(float(width), (xc + bw / 2) * width)
        y2 = min(float(height), (yc + bh / 2) * height)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box in {label_path}: {line}")

        boxes.append([x1, y1, x2, y2])
        labels.append(class_id + 1)  # 0 is reserved for background

    if not boxes:
        raise RuntimeError(
            f"The first image has no boxes. Choose another image for the smoke test."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    print("Test image:", image_path)
    print("Image dimensions:", image.size)
    print("Boxes in test image:", len(boxes))

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES
    )
    model.to(device)
    model.train()

    image_tensor = pil_to_tensor(image).float() / 255.0
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, dtype=torch.int64, device=device),
    }

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )

    optimizer.zero_grad(set_to_none=True)
    losses = model([image_tensor.to(device)], [target])
    total_loss = sum(losses.values())
    total_loss.backward()
    optimizer.step()

    print("\nSmoke test passed!")
    print("Loss components:", {
        key: round(value.item(), 4)
        for key, value in losses.items()
    })
    print("Total loss:", round(total_loss.item(), 4))
    print("No training checkpoint was saved or overwritten.")


if __name__ == "__main__":
    main()
