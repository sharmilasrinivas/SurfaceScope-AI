
from pathlib import Path
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "neu_yolo"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def validate_split(split):
    image_dir = DATA_DIR / "images" / split
    label_dir = DATA_DIR / "labels" / split

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    errors = []
    total_boxes = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            errors.append(f"Missing label: {image_path.name}")
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size

            lines = label_path.read_text(
                encoding="utf-8"
            ).strip().splitlines()

            if not lines or lines == [""]:
                errors.append(f"Empty labels: {image_path.name}")
                continue

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()

                if len(parts) != 5:
                    errors.append(
                        f"Invalid format: {label_path.name}:{line_number}"
                    )
                    continue

                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                valid = (
                    0 <= class_id < len(CLASSES)
                    and 0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                    and x - w / 2 >= -1e-6
                    and y - h / 2 >= -1e-6
                    and x + w / 2 <= 1 + 1e-6
                    and y + h / 2 <= 1 + 1e-6
                )

                if not valid:
                    errors.append(
                        f"Invalid label values: "
                        f"{label_path.name}:{line_number}"
                    )

                total_boxes += 1

        except Exception as exc:
            errors.append(f"{image_path.name}: {exc}")

    print(f"\n--- {split.upper()} ---")
    print("Images:", len(images))
    print("Bounding boxes:", total_boxes)
    print("Errors:", len(errors))

    for error in errors[:20]:
        print(error)

    return images, errors


train_images, train_errors = validate_split("train")
val_images, val_errors = validate_split("validation")

if train_errors or val_errors:
    raise RuntimeError(
        "Dataset validation failed. Fix errors before training."
    )

print("\nAll generated YOLO labels passed structural validation.")


# Visualize six random training examples.
rng = random.Random(42)
samples = rng.sample(train_images, min(6, len(train_images)))

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for ax, image_path in zip(axes.flat, samples):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    ax.imshow(image)

    label_path = (
        DATA_DIR / "labels" / "train"
        / f"{image_path.stem}.txt"
    )

    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, x, y, w, h = line.split()
        class_id = int(class_id)
        x, y, w, h = map(float, (x, y, w, h))

        left = (x - w / 2) * width
        top = (y - h / 2) * height
        box_width = w * width
        box_height = h * height

        rect = patches.Rectangle(
            (left, top),
            box_width,
            box_height,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )

        ax.add_patch(rect)
        ax.text(
            left,
            max(0, top - 3),
            CLASSES[class_id],
            color="white",
            fontsize=7,
            bbox={"facecolor": "red", "alpha": 0.7},
        )

    ax.set_title(image_path.name)
    ax.axis("off")

plt.tight_layout()
plt.show()
