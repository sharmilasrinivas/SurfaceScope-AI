
from pathlib import Path
import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "data" / "processed" / "neu_yolo" / "data.yaml"
CHECKPOINT = (
    ROOT / "runs" / "faster_rcnn" / "smoke_test_metrics_v2"
    / "best_model.pth"
)
CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dataset_root = Path(cfg["path"])
if not dataset_root.is_absolute():
    dataset_root = (ROOT / dataset_root).resolve()

image_dir = dataset_root / cfg["val"]
label_dir = dataset_root / "labels" / image_dir.name
images = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)[:4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(
    weights=None, weights_backbone=None, min_size=320, max_size=320
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, len(CLASS_NAMES) + 1
)

state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
if isinstance(state, dict) and "model" in state:
    state = state["model"]
model.load_state_dict(state)
model.to(device).eval()

print("Device:", device)
print("Images inspected:", len(images))

with torch.no_grad():
    for path in images:
        image = Image.open(path).convert("RGB")
        tensor = to_tensor(image).to(device)
        pred = model([tensor])[0]

        label_path = label_dir / f"{path.stem}.txt"
        gt_ids = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    gt_ids.append(int(parts[0]))

        print(f"\nImage: {path.name}")
        print("Ground-truth classes:", [
            CLASS_NAMES[i] for i in sorted(set(gt_ids))
        ])
        print("Raw predictions:", len(pred["scores"]))

        for threshold in (0.05, 0.25, 0.50):
            mask = pred["scores"] >= threshold
            labels = pred["labels"][mask].cpu().tolist()
            scores = pred["scores"][mask].cpu().tolist()
            print(f"Predictions >= {threshold}: {len(scores)}")
            for cls_id, score in zip(labels[:10], scores[:10]):
                name = (
                    CLASS_NAMES[cls_id - 1]
                    if 1 <= cls_id <= len(CLASS_NAMES)
                    else f"invalid_class_{cls_id}"
                )
                print(f"  {name}: {score:.3f}")
