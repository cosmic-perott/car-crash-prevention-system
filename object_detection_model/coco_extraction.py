import os
import requests
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil

SAVE_DIR = "yolo_car_200"
IMG_DIR = os.path.join(SAVE_DIR, "images")
LBL_DIR = os.path.join(SAVE_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

CAR_CLASS_ID = 2   # COCO: person=1, car=2

ann_file = "instances_train2017.json"

if not os.path.exists(ann_file):
    print("Downloading annotations only...")
    url = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    r = requests.get(url, stream=True)
    open("coco_ann.zip", "wb").write(r.content)

    import zipfile
    with zipfile.ZipFile("coco_ann.zip", "r") as zip_ref:
        zip_ref.extractall(".")

coco = COCO(ann_file)

img_ids = coco.getImgIds(catIds=[CAR_CLASS_ID])
img_ids = img_ids[:200]  # LIMIT TO 200

print(f"Found {len(img_ids)} images with cars")

img_url_prefix = "http://images.cocodataset.org/train2017/"

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]

    img_path = os.path.join(IMG_DIR, file_name)
    lbl_path = os.path.join(LBL_DIR, file_name.replace(".jpg", ".txt"))

    if not os.path.exists(img_path):
        img_url = img_url_prefix + file_name
        r = requests.get(img_url, stream=True)
        with open(img_path, "wb") as f:
            f.write(r.content)

    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[CAR_CLASS_ID])
    anns = coco.loadAnns(ann_ids)

    h = img_info["height"]
    w = img_info["width"]

    with open(lbl_path, "w") as f:
        for ann in anns:
            x, y, bw, bh = ann["bbox"]

            # convert to YOLO format
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h

            f.write(f"0 {x_center} {y_center} {bw} {bh}\n")
