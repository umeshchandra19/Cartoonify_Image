import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pytest
from src.classic import adjust_brightness, cartoonize, denoise, upscale


def _img(h=96, w=128):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_cartoonize_keeps_shape_and_dtype():
    out = cartoonize(_img()); assert out.shape == (96, 128, 3) and out.dtype == np.uint8

def test_cartoonize_rejects_gray():
    with pytest.raises(ValueError): cartoonize(np.zeros((10, 10), np.uint8))

def test_upscale_doubles_size():
    assert upscale(_img(), 2).shape[:2] == (192, 256)

def test_denoise_shape():
    assert denoise(_img()).shape == (96, 128, 3)

def test_brightness_increases_mean():
    img = np.full((8, 8, 3), 100, np.uint8); assert adjust_brightness(img, 30).mean() > img.mean()
