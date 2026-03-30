import cv2
import numpy as np
import time
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found or path incorrect: {path}")
    return img.astype(np.float32) / 255.0
def dark_channel(image, size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark
def estimate_atmospheric_light(image, dark):
    h, w = dark.shape
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))

    dark_vec = dark.reshape(num_pixels)
    image_vec = image.reshape(num_pixels, 3)

    indices = dark_vec.argsort()[-num_brightest:]

    A = np.mean(image_vec[indices], axis=0)
    return A
def estimate_transmission(image, A, size=15, omega=0.95):
    norm_img = image / A
    transmission = 1 - omega * dark_channel(norm_img, size)
    return transmission
def guided_filter(I, p, r=40, eps=1e-3):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q
def refine_transmission(image, transmission):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64) / 255.0
    refined = guided_filter(gray, transmission)
    return refined
def recover(image, transmission, A, t0=0.1):
    transmission = np.maximum(transmission, t0)
    J = np.zeros_like(image)

    for i in range(3):
        J[:, :, i] = (image[:, :, i] - A[i]) / transmission + A[i]

    return np.clip(J, 0, 1)
def dehaze(image_path, output_path="output.png"):
    image = load_image(image_path)

    dark = dark_channel(image)
    A = estimate_atmospheric_light(image, dark)
    transmission = estimate_transmission(image, A)
    transmission_refined = refine_transmission(image, transmission)

    result = recover(image, transmission_refined, A)

    result_uint8 = (result * 255).astype(np.uint8)
    cv2.imwrite(output_path, result_uint8)

    print(f"Saved dehazed image to {output_path}")
    return result_uint8
if __name__ == "__main__":
    start = time.time()
    input_path = "input.jpg"
    output = dehaze(input_path)
    end = time.time()
    print(end-start)
