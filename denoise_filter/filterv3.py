import cv2
import numpy as np

def dehaze_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    gamma_corrected = cv2.LUT(enhanced, table)

    # 5. Edge-preserving smoothing
    final = cv2.bilateralFilter(gamma_corrected, 9, 75, 75)

    return final


img = cv2.imread("2.png")

output = dehaze_image(img)

cv2.imwrite("dehazed_output.jpg", output)
cv2.imshow("Dehazed", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
