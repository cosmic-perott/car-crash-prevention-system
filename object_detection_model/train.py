from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="yolo_car_1000/dataset.yaml",
    epochs=100,      
    imgsz=640,
    batch=8,          
)
