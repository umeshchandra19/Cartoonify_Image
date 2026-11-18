"""Train the cartoon GAN.

    python -m src.train --photos data/photos --cartoons data/cartoons --epochs 20 --steps 500

Photos and cartoon frames are UNPAIRED (two separate folders).
Losses: LSGAN adversarial + content loss (VGG19 features of input vs output, so the picture keeps its structure).
"""
import argparse, os
import tensorflow as tf
from tensorflow.keras.applications import VGG19

from src.data import make_dataset
from src.gan_model import build_discriminator, build_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="data/photos"); ap.add_argument("--cartoons", default="data/cartoons")
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--content_w", type=float, default=10.0); ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    G, D = build_generator(a.size), build_discriminator(a.size)
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=(a.size, a.size, 3))
    feat = tf.keras.Model(vgg.input, vgg.get_layer("block4_conv4").output); feat.trainable = False
    pre = lambda x: tf.keras.applications.vgg19.preprocess_input((x + 1.0) * 127.5)

    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5); d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photos, cartoons = iter(make_dataset(a.photos, a.size, a.batch)), iter(make_dataset(a.cartoons, a.size, a.batch))

    @tf.function
    def step(p, c):
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = G(p, training=True)
            d_real, d_fake = D(c, training=True), D(fake, training=True)
            d_loss = tf.reduce_mean((d_real - 1) ** 2) + tf.reduce_mean(d_fake ** 2)
            adv = tf.reduce_mean((d_fake - 1) ** 2)
            content = tf.reduce_mean(tf.abs(feat(pre(p)) - feat(pre(fake))))
            g_loss = adv + a.content_w * content
        g_opt.apply_gradients(zip(gt.gradient(g_loss, G.trainable_variables), G.trainable_variables))
        d_opt.apply_gradients(zip(dt.gradient(d_loss, D.trainable_variables), D.trainable_variables))
        return g_loss, d_loss

    for ep in range(1, a.epochs + 1):
        for s in range(a.steps):
            g, d = step(next(photos), next(cartoons))
        print(f"epoch {ep}/{a.epochs}  g_loss={float(g):.3f}  d_loss={float(d):.3f}", flush=True)
        G.save_weights(f"{a.out}/generator.weights.h5")


if __name__ == "__main__":
    main()
