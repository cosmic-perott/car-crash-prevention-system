import cv2
import numpy as np

#change this part in order to change the image being noise reduced
IMAGE_URL = "foggy.jpg"


#use of standard Dark Channel Prior (DCP) Dehazing algorithm
def dark_channel(img, size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def estimate_atmosphere(img, dark):
    flat_img = img.reshape(-1, 3)
    flat_dark = dark.ravel()
    indices = np.argsort(flat_dark)[-int(0.001 * len(flat_dark)):]
    return np.mean(flat_img[indices], axis=0)

def transmission_estimate(img, A, omega=0.95, size=15):
    norm_img = img / A
    return 1 - omega * dark_channel(norm_img, size)

def recover(img, t, A, t0=0.1):
    t = np.clip(t, t0, 1)
    J = (img - A) / t[..., None] + A
    return np.clip(J, 0, 1)

def dehaze(image):
    img = image.astype(np.float32) / 255.0
    dark = dark_channel(img)
    A = estimate_atmosphere(img, dark)
    t = transmission_estimate(img, A)
    result = recover(img, t, A)
    return (result * 255).astype(np.uint8)

#contrast enhancement
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)


#edge enhancement
def enhance_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend edges into image
    enhanced = cv2.addWeighted(image, 0.85, edges_colored, 0.15, 0)

    return enhanced


#headlight control
def fix_headlights(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Detect very bright pixels
    mask = v > 230

    # Reduce only brightness slightly (preserve color!)
    v[mask] = v[mask] * 0.75

    hsv_fixed = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_fixed, cv2.COLOR_HSV2BGR)

#sharpening
def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


#main processing stage = getting everything together
def process_image(image_path):
    img = cv2.imread(image_path)

    # Step 1: Dehaze
    dehazed = dehaze(img)

    # Step 2: Contrast boost
    contrast = enhance_contrast(dehazed)

    # Step 3: Fix headlights (important BEFORE sharpening)
    # lights_fixed = fix_headlights(contrast)

    # Step 4: Edge enhancement
    # edges = enhance_edges(lights_fixed)
    edges = enhance_edges(contrast)

    # Step 5: Sharpen
    final = sharpen(edges)

    return img, final



if __name__ == "__main__":
    input_path = IMAGE_URL

    original, result = process_image(input_path)

    cv2.imshow("Original", original)
    cv2.imshow("Enhanced", result)

    cv2.imwrite("enhanced_result.jpg", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
