import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = ["car"]
NUM_SAMPLES = 200
EXPORT_DIR = "yolo_car_200"

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=NUM_SAMPLES,
    shuffle=True,
)

dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
)

print(EXPORT_DIR)
