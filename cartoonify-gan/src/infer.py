"""Run the trained generator on one image."""
import argparse
import cv2, numpy as np
from src.gan_model import build_generator


def load_generator(weights: str, size: int = 256):
    g = build_generator(size); g.load_weights(weights); return g


def cartoonize_gan(gen, img_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (size, size)).astype("float32") / 127.5 - 1.0
    y = gen(x[None], training=False).numpy()[0]
    y = ((y + 1) * 127.5).clip(0, 255).astype("uint8")
    return cv2.resize(cv2.cvtColor(y, cv2.COLOR_RGB2BGR), (w, h))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("out")
    ap.add_argument("--weights", default="checkpoints/generator.weights.h5"); a = ap.parse_args()
    cv2.imwrite(a.out, cartoonize_gan(load_generator(a.weights), cv2.imread(a.image)))
