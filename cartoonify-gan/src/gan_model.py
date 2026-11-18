"""CartoonGAN-style generator and PatchGAN discriminator in TensorFlow/Keras.

Design (kept small so it can train on a normal GPU):
  Generator     : conv stem -> 2 downsampling convs -> N residual blocks -> 2 upsampling convs -> tanh RGB
  Discriminator : PatchGAN (judges overlapping patches as real-cartoon / fake)
Inputs are scaled to [-1, 1].
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def _res_block(x, filters):
    y = layers.Conv2D(filters, 3, padding="same")(x)
    y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same")(y)
    y = layers.BatchNormalization()(y)
    return layers.Add()([x, y])


def build_generator(size: int = 256, n_res: int = 6) -> Model:
    inp = layers.Input((size, size, 3))
    x = layers.Conv2D(64, 7, padding="same")(inp); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    for f in (128, 256):
        x = layers.Conv2D(f, 3, strides=2, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    for _ in range(n_res):
        x = _res_block(x, 256)
    for f in (128, 64):
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    out = layers.Conv2D(3, 7, padding="same", activation="tanh")(x)
    return Model(inp, out, name="generator")


def build_discriminator(size: int = 256) -> Model:
    inp = layers.Input((size, size, 3))
    x = inp
    for f in (64, 128, 256):
        x = layers.Conv2D(f, 4, strides=2, padding="same")(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2D(1, 3, padding="same")(x)  # patch logits, no sigmoid (LSGAN)
    return Model(inp, out, name="discriminator")
