"""Classic OpenCV cartoon filter + the 'Image-hub' helpers (denoise, upscale, brightness).

Steps (same idea as the paper's 'Output' section):
  1. grayscale -> 2. smooth -> 3. detect edges -> 4. smooth colours -> 5. mask colours with edges
"""
import cv2
import numpy as np


def denoise(img: np.ndarray, strength: int = 7) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def upscale(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def adjust_brightness(img: np.ndarray, beta: int = 0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def cartoonize(img: np.ndarray, line_size: int = 7, blur_value: int = 7, bilateral_passes: int = 5) -> np.ndarray:
    """Turn a BGR uint8 image into a cartoon-style BGR uint8 image."""
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected a 3-channel BGR image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    color = img
    for _ in range(bilateral_passes):  # repeated light smoothing keeps edges but flattens colour areas
        color = cv2.bilateralFilter(color, d=7, sigmaColor=40, sigmaSpace=40)
    return cv2.bitwise_and(color, color, mask=edges)
