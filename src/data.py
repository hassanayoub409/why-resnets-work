"""CIFAR-10 loading, preprocessing and input pipelines."""
import numpy as np
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def normalize(images):
    """uint8 (or 0-255 float) images -> float32, per-channel standardized."""
    return ((np.asarray(images, dtype=np.float32) / 255.0) - MEAN) / STD


def load_cifar10():
    """Downloads CIFAR-10 on first use (Keras caches it)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return x_train, y_train.ravel(), x_test, y_test.ravel()


def fake_cifar(n_train=1500, n_test=300, seed=0):
    """A tiny learnable stand-in (each class = a colour + noise). ONLY for smoke tests, never for results."""
    rng = np.random.default_rng(seed)
    colours = rng.uniform(40, 215, size=(10, 3))

    def make(n):
        y = rng.integers(0, 10, n)
        x = colours[y][:, None, None, :] + rng.normal(0, 30, (n, 32, 32, 3))
        return np.clip(x, 0, 255).astype(np.uint8), y
    (x_train, y_train), (x_test, y_test) = make(n_train), make(n_test)
    return x_train, y_train, x_test, y_test


def split_validation(x, y, n_val=5000, seed=0):
    order = np.random.default_rng(seed).permutation(len(x))
    val, train = order[:n_val], order[n_val:]
    return x[train], y[train], x[val], y[val]


def _augment(image, label):
    """Random crop (4 px padding) and horizontal flip: the standard CIFAR recipe."""
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    return tf.image.random_flip_left_right(image), label


def make_dataset(x, y, batch_size, train):
    ds = tf.data.Dataset.from_tensor_slices((normalize(x), y.astype("int32")))
    if train:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True).map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
