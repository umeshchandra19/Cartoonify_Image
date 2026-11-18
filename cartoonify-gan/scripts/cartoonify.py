"""CLI for the classic filter (no GPU, no training needed).

    python scripts/cartoonify.py image in.jpg out.jpg [--denoise] [--upscale 2] [--brightness 10]
    python scripts/cartoonify.py video in.mp4 out.mp4
    python scripts/cartoonify.py gif   in.mp4 out.gif
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from src.classic import adjust_brightness, cartoonize, denoise, upscale
from src.video import process_video, video_to_gif


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("image"); i.add_argument("src"); i.add_argument("dst")
    i.add_argument("--denoise", action="store_true"); i.add_argument("--upscale", type=float, default=1.0)
    i.add_argument("--brightness", type=int, default=0)
    v = sub.add_parser("video"); v.add_argument("src"); v.add_argument("dst")
    g = sub.add_parser("gif"); g.add_argument("src"); g.add_argument("dst")
    a = ap.parse_args()
    if a.cmd == "image":
        img = cv2.imread(a.src)
        if img is None: sys.exit(f"cannot read {a.src}")
        if a.denoise: img = denoise(img)
        if a.upscale != 1.0: img = upscale(img, a.upscale)
        img = adjust_brightness(cartoonize(img), a.brightness)
        cv2.imwrite(a.dst, img); print("saved", a.dst)
    elif a.cmd == "video":
        print(process_video(a.src, a.dst, cartoonize), "frames written")
    else:
        print(video_to_gif(a.src, a.dst), "frames in gif")


if __name__ == "__main__":
    main()
