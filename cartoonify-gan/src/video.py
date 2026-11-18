"""Frame-by-frame video cartoonization, audio-free output, and video->GIF."""
import cv2
import imageio.v2 as imageio


def process_video(src: str, dst: str, fn, max_frames: int | None = None) -> int:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        out.write(fn(frame)); n += 1
    cap.release(); out.release()
    return n


def video_to_gif(src: str, dst: str, every_nth: int = 2, width: int = 320) -> int:
    cap = cv2.VideoCapture(src)
    fps = (cap.get(cv2.CAP_PROP_FPS) or 24) / every_nth
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every_nth == 0:
            h = int(f.shape[0] * width / f.shape[1])
            frames.append(cv2.cvtColor(cv2.resize(f, (width, h)), cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    imageio.mimsave(dst, frames, duration=1000 / fps / 1000, loop=0)
    return len(frames)
