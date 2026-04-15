import cv2
import numpy as np

def dehaze_strong(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)

    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    smooth = cv2.bilateralFilter(img, 9, 100, 100)

    sharp = cv2.addWeighted(img, 1.6, smooth, -0.6, 0)

    return sharp


img = cv2.imread("2.png")

if img is None:
    raise ValueError("Image not found!")

output = dehaze_strong(img)

cv2.imwrite("dehazed_strong.jpg", output)
cv2.imshow("Strong Dehaze", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
