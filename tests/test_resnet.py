"""Run from the repo root:  python -m pytest -q"""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src import data
from src.analysis import gradcam, layer_gradient_norms, load_weights_npz, save_weights_npz
from src.resnet import build_model, convolutional_block, depth_of, identity_block, main_path_convs
from train import WarmupCosine


def test_depth_formula_and_names():
    assert depth_of(6) == 56 and depth_of(2) == 20 and depth_of(12) == 110
    assert build_model(2, skip=True).name == "resnet20"
    assert build_model(2, skip=False).name == "plain20"


def test_skip_flag_only_removes_the_shortcuts():
    res, plain = build_model(2, skip=True), build_model(2, skip=False)
    count = lambda m, kind: sum(isinstance(l, kind) for l in m.layers)
    assert count(res, layers.Add) == 6 and count(plain, layers.Add) == 0     # 3 stages x 2 blocks
    assert main_path_convs(res) == main_path_convs(plain)                    # same main-path layers
    assert len(main_path_convs(res)) == 3 * 3 * 2 + 1                        # 9 per stage... + stem
    assert res.output_shape == plain.output_shape == (None, 10)


def test_block_shapes():
    x = keras.Input((8, 8, 16))
    same = identity_block(layers.Conv2D(16, 1)(x), 3, [4, 4, 16])
    assert tuple(same.shape) == (None, 8, 8, 16)
    down = convolutional_block(x, 3, [8, 8, 32], s=2)
    assert tuple(down.shape) == (None, 4, 4, 32)


def test_model_outputs_probabilities():
    model = build_model(1)
    p = model.predict(np.random.default_rng(0).normal(size=(5, 32, 32, 3)).astype("float32"), verbose=0)
    assert p.shape == (5, 10) and np.allclose(p.sum(axis=1), 1, atol=1e-4)


def test_skip_connection_makes_a_block_pass_its_input_through():
    """If the main path outputs zeros, the skip connection returns the input: the 'easy identity' idea."""
    model = build_model(2, skip=True)
    bn = model.get_layer("s1b1_c_bn")
    bn.gamma.assign(np.zeros(bn.gamma.shape))
    bn.beta.assign(np.zeros(bn.beta.shape))
    probe = keras.Model(model.inputs, [model.get_layer("s1b0_out").output, model.get_layer("s1b1_out").output])
    before, after = probe(np.random.default_rng(1).normal(size=(2, 32, 32, 3)).astype("float32"), training=False)
    assert np.allclose(before.numpy(), after.numpy(), atol=1e-5)   # block 1 changed nothing

    plain = build_model(2, skip=False)
    bn = plain.get_layer("s1b1_c_bn")
    bn.gamma.assign(np.zeros(bn.gamma.shape))
    bn.beta.assign(np.zeros(bn.beta.shape))
    probe = keras.Model(plain.inputs, plain.get_layer("s1b1_out").output)
    assert np.allclose(probe(np.ones((1, 32, 32, 3), "float32"), training=False).numpy(), 0)  # plain: signal destroyed


def test_weights_roundtrip_gives_identical_predictions(tmp_path):
    a = build_model(1)
    a.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
    a.fit(np.random.default_rng(0).normal(size=(16, 32, 32, 3)).astype("float32"), np.arange(16) % 10,
          epochs=1, verbose=0)  # one step so the batch-norm statistics are not at their defaults
    path = tmp_path / "w.npz"
    save_weights_npz(a, path)
    b = build_model(1)
    load_weights_npz(b, path)
    x = np.random.default_rng(2).normal(size=(3, 32, 32, 3)).astype("float32")
    assert np.allclose(a.predict(x, verbose=0), b.predict(x, verbose=0), atol=1e-6)


def test_loading_weights_into_a_different_architecture_fails(tmp_path):
    path = tmp_path / "w.npz"
    save_weights_npz(build_model(1), path)
    with pytest.raises(ValueError):
        load_weights_npz(build_model(2), path)


def test_gradient_measurement_does_not_change_the_model():
    model = build_model(1)
    x, y = np.random.default_rng(0).normal(size=(8, 32, 32, 3)).astype("float32"), np.arange(8) % 10
    before = [w.copy() for w in model.get_weights()]
    norms = layer_gradient_norms(model, x, y)
    assert len(norms) == len(main_path_convs(model)) and all(n > 0 for n in norms)
    assert all(np.array_equal(b, a) for b, a in zip(before, model.get_weights()))


def test_gradcam_heatmap_is_normalised():
    model = build_model(1)
    cam, cls = gradcam(model, np.random.default_rng(0).normal(size=(32, 32, 3)).astype("float32"))
    assert cam.shape == (8, 8) and 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6 and 0 <= cls < 10


def test_warmup_cosine_schedule():
    s = WarmupCosine(peak=0.1, warmup_steps=10, total_steps=110)
    assert float(s(0)) < 0.02 and np.isclose(float(s(10)), 0.1, atol=1e-6)
    assert float(s(60)) < float(s(10)) and float(s(110)) < 1e-6


def test_data_helpers():
    x_tr, y_tr, x_te, y_te = data.fake_cifar(200, 50)
    assert x_tr.shape == (200, 32, 32, 3) and y_te.shape == (50,)
    z = data.normalize(x_tr)
    assert abs(float(z.mean())) < 1.5 and z.dtype == np.float32
    xb, yb = next(iter(data.make_dataset(x_tr, y_tr, 32, train=True)))
    assert tuple(xb.shape) == (32, 32, 32, 3) and tuple(yb.shape) == (32,)
