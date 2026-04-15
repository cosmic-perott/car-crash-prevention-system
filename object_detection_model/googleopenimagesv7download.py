import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = ["Car"]      
TRAIN_SAMPLES = 3000
VAL_SAMPLES = 15000
EXPORT_DIR = "yolo_car_dataset"

train_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=TRAIN_SAMPLES,
    shuffle=True,
)

val_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=CLASSES,
    max_samples=VAL_SAMPLES,
    shuffle=True,
)

train_dataset = train_dataset.filter_labels(
    "ground_truth",
    lambda l: l.confidence is None or l.confidence > 0.5
)

val_dataset = val_dataset.filter_labels(
    "ground_truth",
    lambda l: l.confidence is None or l.confidence > 0.5
)


train_dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",
)

val_dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    split="val",
)

print(EXPORT_DIR)
