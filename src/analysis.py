"""Tools for looking inside a trained network: gradient sizes per layer, Grad-CAM, portable weights."""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .resnet import main_path_convs


# ---------------------------------------------------------------- gradients per layer
def layer_gradient_norms(model, x, y):
    """Size (L2 norm) of the loss gradient for every main-path convolution, in depth order.

    Uses the data loss only (no weight-decay term). The forward pass in training mode would change the
    batch-norm moving averages, so the weights are restored afterwards and the model is left untouched.
    """
    kernels = [model.get_layer(name).kernel for name in main_path_convs(model)]
    saved = model.get_weights()
    with tf.GradientTape() as tape:
        probs = model(x, training=True)
        loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, probs))
    grads = tape.gradient(loss, kernels)
    model.set_weights(saved)
    return [float(tf.norm(g)) for g in grads]


# ---------------------------------------------------------------- Grad-CAM
def gradcam(model, x, class_index=None):
    """Where did the network look? Returns (heatmap in [0, 1] shaped like the last feature map, class index).

    Grad-CAM: weight each channel of the last feature map by how much it raises the chosen class score.
    """
    last = [l for l in model.layers if l.name.endswith("_out")][-1]
    probe = keras.Model(model.inputs, [last.output, model.output])
    x = tf.convert_to_tensor(x[None] if x.ndim == 3 else x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        features, probs = probe(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(probs[0]))
        score = probs[:, class_index]
    grads = tape.gradient(score, features)
    channel_weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(features * channel_weights, axis=-1))[0].numpy()
    peak = cam.max()
    return (cam / peak if peak > 0 else cam), class_index


# ---------------------------------------------------------------- portable weights
def _key(layer, variable):
    """Stable key independent of the Keras/TensorFlow version: 'layer name | variable name'."""
    return f"{layer.name}|{variable.name.split('/')[-1].split(':')[0]}"


def save_weights_npz(model, path):
    np.savez_compressed(path, **{_key(l, v): v.numpy() for l in model.layers for v in l.weights})


def load_weights_npz(model, path):
    data = np.load(path)
    missing = []
    for layer in model.layers:
        for variable in layer.weights:
            key = _key(layer, variable)
            if key not in data.files:
                missing.append(key)
            elif tuple(data[key].shape) != tuple(variable.shape):
                raise ValueError(f"Shape mismatch for {key}: file {data[key].shape} vs model {tuple(variable.shape)}")
            else:
                variable.assign(data[key])
    if missing:
        raise ValueError(f"{len(missing)} weights missing in {path} (e.g. {missing[0]}). "
                         "Was the file trained with the same --blocks setting?")
