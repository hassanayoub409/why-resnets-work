"""ResNet building blocks and a configurable ResNet / plain-network builder (TensorFlow / Keras).

The two block types follow the Deep Learning Specialization's ResNet lab (bottleneck design):

    identity block      : CONV 1x1 -> BN -> ReLU -> CONV fxf -> BN -> ReLU -> CONV 1x1 -> BN -> (+ input) -> ReLU
    convolutional block : same main path, but the shortcut goes through CONV 1x1 -> BN so the shapes match

Passing skip=False removes the shortcut, which turns the very same network into a "plain" deep network.
This is what makes a fair comparison possible: identical layers, only the skip connections differ.
"""
from tensorflow import keras
from tensorflow.keras import layers

CIFAR_STAGE_WIDTHS = (16, 32, 64)  # bottleneck width per stage; each block outputs 4x this many channels


def _l2(weight_decay):
    return keras.regularizers.l2(weight_decay) if weight_decay else None


def _conv_bn(x, filters, kernel, strides=1, padding="same", name=None, relu=True, wd=0.0):
    x = layers.Conv2D(filters, kernel, strides=strides, padding=padding, use_bias=False,
                      kernel_initializer="he_normal", kernel_regularizer=_l2(wd), name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x) if relu else x


def identity_block(x, f, filters, skip=True, name="id", wd=0.0):
    """Input and output have the same shape, so the input can be added straight back."""
    f1, f2, f3 = filters
    shortcut = x
    x = _conv_bn(x, f1, 1, padding="valid", name=f"{name}_a", wd=wd)
    x = _conv_bn(x, f2, f, padding="same", name=f"{name}_b", wd=wd)
    x = _conv_bn(x, f3, 1, padding="valid", name=f"{name}_c", relu=False, wd=wd)
    if skip:
        x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(x)


def convolutional_block(x, f, filters, s=2, skip=True, name="conv", wd=0.0):
    """Shape changes (more channels and/or stride s), so the shortcut needs its own 1x1 convolution."""
    f1, f2, f3 = filters
    shortcut = x
    x = _conv_bn(x, f1, 1, strides=s, padding="valid", name=f"{name}_a", wd=wd)
    x = _conv_bn(x, f2, f, padding="same", name=f"{name}_b", wd=wd)
    x = _conv_bn(x, f3, 1, padding="valid", name=f"{name}_c", relu=False, wd=wd)
    if skip:
        shortcut = layers.Conv2D(f3, 1, strides=s, padding="valid", use_bias=False,
                                 kernel_initializer="he_normal", kernel_regularizer=_l2(wd),
                                 name=f"{name}_short_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_short_bn")(shortcut)
        x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(x)


def depth_of(blocks_per_stage, n_stages=3):
    """Counted like the ResNet papers: stem conv + 3 convs per block + the final dense layer."""
    return 1 + 3 * blocks_per_stage * n_stages + 1


def build_model(blocks_per_stage=6, skip=True, input_shape=(32, 32, 3), classes=10,
                widths=CIFAR_STAGE_WIDTHS, weight_decay=0.0):
    """CIFAR-style network. blocks_per_stage=6 gives 56 layers; 2 gives 20; 12 gives 110."""
    inputs = keras.Input(input_shape, name="image")
    x = _conv_bn(inputs, widths[0], 3, name="stem", wd=weight_decay)
    for stage, w in enumerate(widths):
        x = convolutional_block(x, 3, [w, w, 4 * w], s=1 if stage == 0 else 2, skip=skip,
                                name=f"s{stage + 1}b0", wd=weight_decay)
        for b in range(1, blocks_per_stage):
            x = identity_block(x, 3, [w, w, 4 * w], skip=skip, name=f"s{stage + 1}b{b}", wd=weight_decay)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    outputs = layers.Dense(classes, activation="softmax", kernel_regularizer=_l2(weight_decay),
                           name="classifier")(x)
    kind = "resnet" if skip else "plain"
    return keras.Model(inputs, outputs, name=f"{kind}{depth_of(blocks_per_stage)}")


def main_path_convs(model):
    """Names of the convolutions on the main path, in depth order (shortcut convolutions excluded)."""
    return [l.name for l in model.layers
            if isinstance(l, layers.Conv2D) and "_short_" not in l.name]
