
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "neu_yolo"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {
    name: index for index, name in enumerate(CLASSES)
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_image_index(image_dir):
    index = {}

    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem.lower(), []).append(path)

    return index


def convert_box(box, width, height):
    xmin = float(box.findtext("xmin"))
    ymin = float(box.findtext("ymin"))
    xmax = float(box.findtext("xmax"))
    ymax = float(box.findtext("ymax"))

    if not (
        0 <= xmin < xmax <= width
        and 0 <= ymin < ymax <= height
    ):
        raise ValueError(
            f"Invalid box: {(xmin, ymin, xmax, ymax)} "
            f"for image size {(width, height)}"
        )

    # Pascal VOC coordinates -> normalized YOLO coordinates.
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return x_center, y_center, box_width, box_height


def prepare_split(split):
    image_dir = RAW_DIR / split / "images"
    annotation_dir = RAW_DIR / split / "annotations"

    output_images = OUTPUT_DIR / "images" / split
    output_labels = OUTPUT_DIR / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index(image_dir)
    xml_files = sorted(annotation_dir.glob("*.xml"))

    stats = {
        "annotations_processed": 0,
        "images_processed": 0,
        "objects_processed": 0,
        "skipped_annotations": [],
    }

    # Rebuild this split to avoid retaining stale output files.
    for folder in [output_images, output_labels]:
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()

        filename = root.findtext("filename", "").strip()
        stem = Path(filename).stem.lower() if filename else ""

        candidates = image_index.get(stem, [])

        # Never guess if an image is missing or ambiguous.
        if len(candidates) != 1:
            stats["skipped_annotations"].append(
                (xml_path.name, filename, len(candidates))
            )
            continue

        source_image = candidates[0]
        image_name = source_image.name
        destination_image = output_images / image_name
        destination_label = output_labels / f"{source_image.stem}.txt"

        try:
            with Image.open(source_image) as image:
                width, height = image.size

            size = root.find("size")
            xml_width = int(size.findtext("width"))
            xml_height = int(size.findtext("height"))

            if (width, height) != (xml_width, xml_height):
                raise ValueError(
                    f"XML/image size mismatch in {xml_path.name}: "
                    f"XML={(xml_width, xml_height)}, "
                    f"image={(width, height)}"
                )

            label_lines = []

            for obj in root.findall("object"):
                class_name = obj.findtext("name", "").strip()

                if class_name not in CLASS_TO_ID:
                    raise ValueError(
                        f"Unknown class {class_name!r} "
                        f"in {xml_path.name}"
                    )

                box = obj.find("bndbox")

                x_center, y_center, box_w, box_h = convert_box(
                    box, width, height
                )

                class_id = CLASS_TO_ID[class_name]

                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{box_w:.6f} {box_h:.6f}"
                )

            if not label_lines:
                raise ValueError(
                    f"No objects found in {xml_path.name}"
                )

            shutil.copy2(source_image, destination_image)
            destination_label.write_text(
                "\n".join(label_lines) + "\n",
                encoding="utf-8",
            )

            stats["annotations_processed"] += 1
            stats["images_processed"] += 1
            stats["objects_processed"] += len(label_lines)

        except Exception as exc:
            # Stop instead of silently producing a partial dataset.
            raise RuntimeError(
                f"Failed to process {xml_path}: {exc}"
            ) from exc

    return stats


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DIR}"
        )

    train_stats = prepare_split("train")
    val_stats = prepare_split("validation")

    # A YAML file tells Ultralytics where images and classes are.
    config = {
        "path": OUTPUT_DIR.resolve().as_posix(),
        "train": "images/train",
        "val": "images/validation",
        "names": {
            index: name for index, name in enumerate(CLASSES)
        },
    }

    config_path = OUTPUT_DIR / "data.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print("\nTRAIN:", train_stats)
    print("\nVALIDATION:", val_stats)
    print("\nYOLO config:", config_path)

    if train_stats["skipped_annotations"]:
        print("Training annotations skipped:")
        for item in train_stats["skipped_annotations"]:
            print(item)

    if val_stats["skipped_annotations"]:
        print("Validation annotations skipped:")
        for item in val_stats["skipped_annotations"]:
            print(item)


if __name__ == "__main__":
    main()
