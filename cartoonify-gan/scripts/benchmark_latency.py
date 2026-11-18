"""Measure per-image inference time so latency claims are backed by YOUR numbers.

    python scripts/benchmark_latency.py --image samples/input.png --runs 50                 # classic filter
    python scripts/benchmark_latency.py --image samples/input.png --gan checkpoints/generator.weights.h5

Run it before and after an optimisation (e.g. smaller generator, mixed precision, TFLite) and compare the medians.
"""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from src.classic import cartoonize


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True)
    ap.add_argument("--runs", type=int, default=30); ap.add_argument("--gan")
    a = ap.parse_args(); img = cv2.imread(a.image)
    fn = cartoonize
    if a.gan:
        from src.infer import cartoonize_gan, load_generator
        gen = load_generator(a.gan); fn = lambda x: cartoonize_gan(gen, x)
    fn(img)  # warm-up
    t = []
    for _ in range(a.runs):
        s = time.perf_counter(); fn(img); t.append((time.perf_counter() - s) * 1000)
    print(f"{'GAN' if a.gan else 'classic'}: median {statistics.median(t):.1f} ms, p95 {sorted(t)[int(.95*len(t))-1]:.1f} ms over {a.runs} runs")


if __name__ == "__main__":
    main()
