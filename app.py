"""Streamlit app: Why ResNets Work. Classify photos with a ResNet and see what skip connections change.

Run with:  streamlit run app.py
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src import data
from src.analysis import gradcam, load_weights_npz
from src.paths import ASSETS_DIR, MODELS_DIR
from src.resnet import build_model

GREEN, AMBER = "#4F7D32", "#F5B52E"
WITH_SKIP, WITHOUT_SKIP = "With skip connections (ResNet)", "Without skip connections (plain)"

st.set_page_config(page_title="Why ResNets Work", page_icon="🧱", layout="wide")


@st.cache_resource(show_spinner="Loading the trained network...")
def load_resources():
    results = json.loads((MODELS_DIR / "results.json").read_text())
    model = build_model(results["config"]["blocks_per_stage"], skip=True)
    load_weights_npz(model, MODELS_DIR / "resnet_weights.npz")
    samples = np.load(MODELS_DIR / "samples.npz")
    return model, results, samples["images"], samples["labels"]


needed = ["results.json", "resnet_weights.npz", "samples.npz"]
if not all((MODELS_DIR / f).exists() for f in needed):
    st.title("🧱 Why ResNets Work")
    st.error("The trained model files are missing from the `models/` folder.")
    st.markdown("Train them for free on Google Colab (about an hour), then copy `resnet_weights.npz`, `results.json` "
                "and `samples.npz` into `models/`. The README has step-by-step instructions "
                "(`notebooks/train_on_colab.ipynb`).")
    st.stop()
model, results, sample_images, sample_labels = load_resources()
classes = results["classes"]
cfg, models_info = results["config"], results["models"]

st.title("🧱 Why ResNets Work")
st.markdown("**What do skip connections change in a deep neural network?** Classify a photo with a 56-layer ResNet, "
            "then compare it with the exact same network *without* its skip connections.")
if cfg.get("fake_data"):
    st.warning("These files come from a quick smoke test on synthetic data, not CIFAR-10. "
               "Train the real model (see the README) before trusting any result.")

tab_try, tab_why, tab_about = st.tabs(["1 · Classify a photo", "2 · Why skip connections?", "About"])


# ------------------------------------------------------------------ helpers
def center_square(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


def overlay(square: Image.Image, heat: np.ndarray, size=256) -> np.ndarray:
    base = np.asarray(square.resize((size, size), Image.BICUBIC)).astype(float)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    a = (0.75 * np.asarray(heat_img).astype(float) / 255.0)[..., None]
    return (base * (1 - a) + np.array([0x4F, 0x7D, 0x32]) * a).astype(np.uint8)


# ------------------------------------------------------------------ 1 · Classify a photo
with tab_try:
    st.markdown(f"This network was trained to tell apart **{len(classes)} kinds of things**: "
                + ", ".join(classes) + ". Try one of its test images, or upload your own photo of one of them.")
    source = st.radio("Image", ["Sample image", "Upload my own"], horizontal=True)
    true_label = None
    if source == "Sample image":
        i = st.selectbox("Choose a sample", range(len(sample_images)),
                         format_func=lambda k: f"Sample {k + 1} (true class: {classes[int(sample_labels[k])]})")
        square = Image.fromarray(sample_images[i])
        true_label = int(sample_labels[i])
    else:
        upload = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg", "webp"])
        square = center_square(Image.open(upload)) if upload else None

    if square is None:
        st.info("Upload a photo to see what the network thinks.")
    else:
        small = np.asarray(square.resize((32, 32), Image.LANCZOS))
        x = data.normalize(small)
        probs = model(x[None], training=False).numpy()[0]
        heat, _ = gradcam(model, x)
        order = np.argsort(-probs)
        c1, c2, c3 = st.columns(3)
        c1.image(np.asarray(square.resize((256, 256), Image.BICUBIC)), caption="Your photo (cropped to a square)")
        c2.image(np.repeat(np.repeat(small, 8, axis=0), 8, axis=1),
                 caption="What the network actually sees (32×32 pixels)")
        c3.image(overlay(square, heat), caption="Where it looked: darker green = more important")

        top = order[0]
        st.markdown(f"### The network's answer: **{classes[top]}** ({probs[top]:.0%} sure)")
        st.markdown(f"Next most likely: {classes[order[1]]} ({probs[order[1]]:.0%}), "
                    f"{classes[order[2]]} ({probs[order[2]]:.0%}).")
        if true_label is not None:
            (st.success if top == true_label else st.error)(
                f"{'Correct' if top == true_label else 'Wrong'}: this image's true class is {classes[true_label]}.")
        if probs[top] < 0.5:
            st.warning("The network is not very sure about this one.")
    with st.expander("Good to know"):
        st.markdown(
            "- The network only ever sees **tiny 32×32 images**, so fine detail is lost.\n"
            "- It can only answer with one of its 10 classes. Show it a person or a banana and it will still pick "
            "one of the 10.\n"
            "- The green map comes from *Grad-CAM*: it highlights the image regions that pushed the network towards "
            "its answer. It is coarse (8×8 cells) but shows whether the network looked at the object or the background."
        )

# ------------------------------------------------------------------ 2 · Why skip connections?
with tab_why:
    st.subheader("The idea")
    st.markdown(
        "In a ResNet, each block adds its **input back to its output** (the green path). If a block has nothing "
        "useful to add, it can simply pass the signal through unchanged, so stacking many blocks is safe. "
        "The two networks compared below have *exactly the same layers*; the only difference is whether the green "
        "shortcut is there."
    )
    st.image(str(ASSETS_DIR / "blocks.png"))

    st.subheader("Does it matter? Same network, with and without skip connections")
    res_i, plain_i = models_info["resnet"], models_info["plain"]
    m1, m2 = st.columns(2)
    m1.metric(WITH_SKIP, f"{res_i['test_accuracy']:.1%}", help="Accuracy on the 10,000 CIFAR-10 test images")
    m2.metric(WITHOUT_SKIP, "failed to train" if plain_i["diverged"] else f"{plain_i['test_accuracy']:.1%}")
    gap = (res_i["test_accuracy"] - plain_i["test_accuracy"]) * 100
    if plain_i["diverged"]:
        st.markdown("The plain network's training broke down completely (its loss became invalid).")
    elif gap > 1:
        st.markdown(f"**The skip connections made a clear difference: {gap:.1f} percentage points** of test accuracy "
                    f"with the same {cfg['depth']} layers and the same training recipe.")
    elif gap < -1:
        st.markdown(f"In this run the plain network was actually **{-gap:.1f} points better**. Skip connections do not "
                    "always help at moderate depth, especially when batch normalization is used.")
    else:
        st.markdown("Both networks ended within one percentage point of each other. At this depth batch normalization "
                    "already makes a plain network trainable; the benefit of skip connections grows with depth.")
    runs = results.get("plain_runs", [])
    if len(runs) > 1:
        st.caption("To be fair to the plain network it was trained with " +
                   " and ".join(f"peak learning rate {r['peak_lr']}" for r in runs) +
                   "; the run with the better validation accuracy is shown.")

    metric = st.radio("Learning curve", ["Validation accuracy", "Training accuracy"], horizontal=True)
    key = "val_accuracy" if metric.startswith("Validation") else "accuracy"
    curves = pd.DataFrame({WITH_SKIP: pd.Series(res_i["history"][key]), WITHOUT_SKIP: pd.Series(plain_i["history"][key])})
    curves.index = np.arange(1, len(curves) + 1)
    curves.index.name = "Epoch"
    st.line_chart(curves, color=[GREEN, AMBER])
    st.caption("An epoch is one pass over the training images. Higher is better.")

    st.subheader("Why? How strongly does each layer learn?")
    st.markdown(
        "A network learns by sending an error signal (the *gradient*) backwards from the answer to every layer. "
        "Below is the size of that signal at each layer, from the first layer (left) to the last (right). "
        "A good signal is reasonably even across layers."
    )
    grads = results["gradients"]
    phase = st.radio("When?", ["Before training", "After training"], horizontal=True)
    which = "start" if phase.startswith("Before") else "end"
    frames = []
    for label, name in ((WITH_SKIP, "resnet"), (WITHOUT_SKIP, "plain")):
        values = grads[name][which]
        if values:
            frames.append(pd.DataFrame({"layer": grads["layers"], "gradient": values, "network": label}))
    if frames:
        chart = (alt.Chart(pd.concat(frames)).mark_line(strokeWidth=3).encode(
            x=alt.X("layer:Q", title="Layer (1 = closest to the input)"),
            y=alt.Y("gradient:Q", scale=alt.Scale(type="log"), title="Size of the gradient (log scale)"),
            color=alt.Color("network:N", scale=alt.Scale(domain=[WITH_SKIP, WITHOUT_SKIP], range=[GREEN, AMBER]),
                            legend=alt.Legend(title=None, orient="bottom")),
        ).properties(width="container", height=320))
        st.altair_chart(chart)

    start_plain, start_res = grads["plain"]["start"], grads["resnet"]["start"]
    if start_plain and start_res:
        spread = lambda v: max(v) / min(v)
        first_vs_last = start_plain[0] / start_plain[-1]
        shape = ("much **larger** near the input than near the output" if first_vs_last > 10 else
                 "much **smaller** near the input than near the output (the classic *vanishing gradient*)"
                 if first_vs_last < 0.1 else "similar at both ends")
        st.markdown(
            f"- At the start of training, gradient sizes in the **plain** network differ by a factor of "
            f"**{spread(start_plain):,.0f}×** between layers, and are {shape}. With **skip connections** the factor is "
            f"**{spread(start_res):,.0f}×**.\n"
            "- You may have read that deep plain networks suffer from vanishing gradients. These networks use batch "
            "normalization, so what we measure is closer to an *imbalance* between layers. Either way, the more uneven "
            "the signal, the harder it is to pick one learning rate that suits every layer."
        )

# ------------------------------------------------------------------ About
with tab_about:
    st.markdown(
        f"""
**The network.** A {cfg['depth']}-layer ResNet built from the two block types of the Deep Learning
Specialization ResNet lab (*identity block* and *convolutional block*, both bottleneck designs) in TensorFlow / Keras,
with a small-image stem, {res_i['params']:,} parameters. The "plain" network has the same layers minus the shortcuts
({plain_i['params']:,} parameters).

**Training.** CIFAR-10: {cfg['train']:,} training, {cfg['validation']:,} validation and {cfg['test']:,} test images.
{cfg['epochs']} epochs, batch size {cfg['batch_size']}, SGD with momentum, peak learning rate {cfg['peak_lr']}
(warm-up then cosine decay), weight decay, random crops and flips. The ResNet took
{res_i['seconds'] / 60:.0f} minutes ({'with' if results['environment']['gpu'] else 'without'} a GPU,
TensorFlow {results['environment']['tensorflow']}).

**Limits.** One training run per network (no repeated seeds), so small differences are not meaningful.
Images are 32×32 and limited to 10 classes.
"""
    )
    per_class = pd.Series(results["resnet_per_class"]).sort_values(ascending=False)
    best, worst = per_class.index[0], per_class.index[-1]
    st.markdown(f"**Easiest class:** {best} ({per_class.iloc[0]:.0%} correct). **Hardest:** {worst} "
                f"({per_class.iloc[-1]:.0%} correct).")
    st.caption(
        "Data: CIFAR-10, A. Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images. "
        "Architecture: K. He, X. Zhang, S. Ren, J. Sun (2015), Deep Residual Learning for Image Recognition. "
        "The concepts were learned in the Convolutional Neural Networks course (DeepLearning.AI on Coursera); "
        "this code is an independent re-implementation and contains no course materials or datasets."
    )
