from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

image_path = "yolo_car_200/images/train/000000567149.jpg" 

results = model(image_path)

annotated = results[0].plot()

cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output.jpg", annotated)

print("Saved as output.jpg")
