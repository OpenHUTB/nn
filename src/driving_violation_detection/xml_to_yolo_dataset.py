import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to a YOLO dataset.")
    parser.add_argument("--input", required=True, help="Directory containing images and XML files.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    return parser.parse_args()


def find_image_for_stem(input_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        candidate = input_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def voc_box_to_yolo(size, box):
    width, height = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = int(size_node.findtext("width", default="0"))
    height = int(size_node.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    labels = []
    class_map = {}

    for obj in root.findall("object"):
        name = obj.findtext("name", default="TrafficSign").strip()
        if name not in class_map:
            class_map[name] = None

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", default="0"))
        ymin = float(bbox.findtext("ymin", default="0"))
        xmax = float(bbox.findtext("xmax", default="0"))
        ymax = float(bbox.findtext("ymax", default="0"))

        if xmax <= xmin or ymax <= ymin:
            continue

        labels.append((name, (xmin, ymin, xmax, ymax)))

    return (width, height), labels


def ensure_dirs(base_output: Path):
    for split in ("train", "val"):
        (base_output / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_output / "labels" / split).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files found in: {input_dir}")

    ensure_dirs(output_dir)

    dataset_items = []
    class_names = []
    class_to_id = {}

    for xml_path in xml_files:
        image_path = find_image_for_stem(input_dir, xml_path.stem)
        if image_path is None:
            print(f"skip: missing image for {xml_path.name}")
            continue

        size, labels = parse_xml(xml_path)
        if not labels:
            print(f"skip: no valid objects in {xml_path.name}")
            continue

        for class_name, _ in labels:
            if class_name not in class_to_id:
                class_to_id[class_name] = len(class_names)
                class_names.append(class_name)

        dataset_items.append((image_path, size, labels))

    if not dataset_items:
        raise ValueError("No valid image/XML pairs found.")

    random.seed(args.seed)
    random.shuffle(dataset_items)

    split_index = int(len(dataset_items) * args.train_ratio)
    if split_index <= 0:
        split_index = 1
    if split_index >= len(dataset_items):
        split_index = len(dataset_items) - 1

    train_items = dataset_items[:split_index]
    val_items = dataset_items[split_index:]

    for split_name, items in (("train", train_items), ("val", val_items)):
        for image_path, size, labels in items:
            target_image = output_dir / "images" / split_name / image_path.name
            target_label = output_dir / "labels" / split_name / f"{image_path.stem}.txt"

            shutil.copy2(image_path, target_image)

            width, height = size
            lines = []
            for class_name, box in labels:
                class_id = class_to_id[class_name]
                x_center, y_center, box_width, box_height = voc_box_to_yolo((width, height), box)
                lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                )

            target_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for class_id, class_name in enumerate(class_names):
        yaml_lines.append(f"  {class_id}: {class_name}")

    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"done: {len(dataset_items)} samples")
    print(f"train: {len(train_items)}")
    print(f"val: {len(val_items)}")
    print(f"classes: {class_names}")
    print(f"dataset yaml: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
