"""tf.data input pipeline with augmentation (random crop + flip + small brightness jitter)."""
import tensorflow as tf


def _load(path, size):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (size + 32, size + 32))
    return img


def _augment(img, size):
    img = tf.image.random_crop(img, (size, size, 3))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 15.0)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img / 127.5 - 1.0


def make_dataset(folder: str, size: int = 256, batch: int = 8):
    files = tf.data.Dataset.list_files(f"{folder}/*", shuffle=True)
    ds = files.map(lambda p: _augment(_load(p, size), size), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(512).batch(batch, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE)
