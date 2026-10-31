
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "processed" / "neu_yolo" / "images" / "validation"
LABEL_DIR = ROOT / "data" / "processed" / "neu_yolo" / "labels" / "validation"

CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_CLASSES = {0, 4}


def main():
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Validation folders not found:\n{IMAGE_DIR}\n{LABEL_DIR}"
        )

    images = {
        image.stem: image
        for image in IMAGE_DIR.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    }

    counts = Counter()
    matches = {class_id: [] for class_id in TARGET_CLASSES}
    missing_images = []

    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        image_path = images.get(label_path.stem)

        if image_path is None:
            missing_images.append(label_path.name)
            continue

        found_classes = set()

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue

            try:
                class_id = int(float(parts[0]))
            except (ValueError, IndexError):
                continue

            counts[class_id] += 1
            if class_id in TARGET_CLASSES:
                found_classes.add(class_id)

        for class_id in found_classes:
            matches[class_id].append(image_path)

    print("\nVALIDATION DATASET AUDIT")
    print("=" * 55)
    print(f"Validation images found: {len(images)}")
    print(f"Annotation files found: {len(list(LABEL_DIR.glob('*.txt')))}")

    print("\nOBJECT COUNTS BY CLASS")
    for class_id, name in CLASS_NAMES.items():
        print(f"{name:20s}: {counts[class_id]} objects")

    for class_id in sorted(TARGET_CLASSES):
        name = CLASS_NAMES[class_id]
        selected = sorted(matches[class_id])

        print(f"\n{name.upper()} — {len(selected)} images")
        print("-" * 55)

        for image_path in selected[:15]:
            print(image_path)

        if len(selected) > 15:
            print(f"... and {len(selected) - 15} more images")

        output_file = ROOT / "reports" / f"validation_{name}_images.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            "\n".join(str(path) for path in selected),
            encoding="utf-8",
        )
        print(f"Full list saved to: {output_file}")

    if missing_images:
        print(f"\nWarning: {len(missing_images)} labels have no matching image.")
        print("First examples:", missing_images[:5])


if __name__ == "__main__":
    main()
