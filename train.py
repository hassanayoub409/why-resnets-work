"""Train a ResNet and the same network WITHOUT skip connections on CIFAR-10, then analyse both.

    python train.py                      # full run (use a GPU, e.g. Google Colab; see README)
    python train.py --fake-data --epochs 2 --blocks 1 --out /tmp/smoke   # seconds-long smoke test, no download

Writes to --out (default models/): resnet_weights.npz, results.json, samples.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import data
from src.analysis import layer_gradient_norms, save_weights_npz
from src.resnet import build_model, depth_of, main_path_convs

WEIGHT_DECAY = 2e-4  # L2 penalty on convolution kernels


class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate rises linearly during warm-up, then follows a cosine curve down to zero."""

    def __init__(self, peak, warmup_steps, total_steps):
        super().__init__()
        self.peak, self.warmup, self.total = float(peak), float(warmup_steps), float(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak * (step + 1.0) / max(self.warmup, 1.0)
        progress = tf.clip_by_value((step - self.warmup) / max(self.total - self.warmup, 1.0), 0.0, 1.0)
        cosine = 0.5 * self.peak * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup, warm, cosine)

    def get_config(self):
        return {"peak": self.peak, "warmup_steps": self.warmup, "total_steps": self.total}


def train_one(name, skip, lr, args, train_ds, val_ds, test_ds, grad_batch, steps_per_epoch):
    keras.utils.set_random_seed(args.seed)  # identical initial conditions for both networks
    model = build_model(args.blocks, skip=skip, weight_decay=WEIGHT_DECAY)
    gx, gy = grad_batch
    grads_start = layer_gradient_norms(model, gx, gy)
    print(f"\n=== {name}: {model.name}, {model.count_params():,} parameters, peak learning rate {lr} ===")

    schedule = WarmupCosine(lr, steps_per_epoch, steps_per_epoch * args.epochs)
    model.compile(optimizer=keras.optimizers.SGD(schedule, momentum=0.9, nesterov=True),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2, shuffle=False,
                        callbacks=[keras.callbacks.TerminateOnNaN()]).history
    seconds = time.time() - start
    history = {k: [float(v) for v in vals] for k, vals in history.items() if k in
               ("loss", "accuracy", "val_loss", "val_accuracy")}
    diverged = bool(np.isnan(history["loss"]).any()) if history["loss"] else True
    grads_end = layer_gradient_norms(model, gx, gy) if not diverged else None
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"{name}: test accuracy {test_acc:.4f}  ({seconds / 60:.1f} min)")
    return model, {
        "depth": depth_of(args.blocks), "params": int(model.count_params()), "history": history,
        "test_accuracy": float(test_acc), "test_loss": float(test_loss),
        "seconds": round(seconds, 1), "diverged": diverged, "peak_lr": lr,
    }, grads_start, grads_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=6, help="blocks per stage: 6 -> 56 layers, 2 -> 20, 12 -> 110")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1, help="peak learning rate")
    ap.add_argument("--plain-lr", type=float, default=0.02,
                    help="extra run of the plain network with this lower peak learning rate (0 = skip); "
                         "the better plain run, judged on validation data, is the one reported")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    ap.add_argument("--fake-data", action="store_true", help="tiny synthetic data, smoke test only")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.fake_data:
        x_train, y_train, x_test, y_test = data.fake_cifar()
        x_train, y_train, x_val, y_val = data.split_validation(x_train, y_train, n_val=300)
    else:
        x_train, y_train, x_test, y_test = data.load_cifar10()
        x_train, y_train, x_val, y_val = data.split_validation(x_train, y_train, n_val=5000)
    print(f"{len(x_train)} train / {len(x_val)} validation / {len(x_test)} test images. "
          f"GPU: {bool(tf.config.list_physical_devices('GPU'))}")

    train_ds = data.make_dataset(x_train, y_train, args.batch_size, train=True)
    val_ds = data.make_dataset(x_val, y_val, args.batch_size, train=False)
    test_ds = data.make_dataset(x_test, y_test, args.batch_size, train=False)
    steps = int(np.ceil(len(x_train) / args.batch_size))
    grad_batch = (data.normalize(x_train[:128]), y_train[:128].astype("int32"))

    results = {"config": {"blocks_per_stage": args.blocks, "depth": depth_of(args.blocks), "epochs": args.epochs,
                          "batch_size": args.batch_size, "peak_lr": args.lr, "weight_decay": WEIGHT_DECAY,
                          "seed": args.seed, "train": len(x_train), "validation": len(x_val), "test": len(x_test),
                          "fake_data": args.fake_data},
               "classes": data.CLASS_NAMES, "models": {}, "gradients": {},
               "environment": {"tensorflow": tf.__version__, "gpu": bool(tf.config.list_physical_devices("GPU"))}}

    runs = [("resnet", True, args.lr), ("plain", False, args.lr)]
    if args.plain_lr and args.plain_lr != args.lr:
        runs.append(("plain (lower learning rate)", False, args.plain_lr))
    outcomes, resnet = {}, None
    for name, skip, lr in runs:
        model, info, g_start, g_end = train_one(name, skip, lr, args, train_ds, val_ds, test_ds, grad_batch, steps)
        outcomes[name] = (info, g_start, g_end)
        if skip:
            resnet = model

    # The plain network gets a fair chance: keep whichever plain run did best on VALIDATION data.
    plain_names = [n for n in outcomes if n != "resnet"]
    chosen = max(plain_names, key=lambda n: (not outcomes[n][0]["diverged"],
                                             outcomes[n][0]["history"]["val_accuracy"][-1]
                                             if outcomes[n][0]["history"]["val_accuracy"] else -1))
    results["models"]["resnet"] = outcomes["resnet"][0]
    results["models"]["plain"] = outcomes[chosen][0]
    results["gradients"]["resnet"] = {"start": outcomes["resnet"][1], "end": outcomes["resnet"][2]}
    results["gradients"]["plain"] = {"start": outcomes[chosen][1], "end": outcomes[chosen][2]}
    results["plain_runs"] = [{"name": n, "peak_lr": outcomes[n][0]["peak_lr"],
                              "final_val_accuracy": (outcomes[n][0]["history"]["val_accuracy"] or [None])[-1],
                              "test_accuracy": outcomes[n][0]["test_accuracy"],
                              "diverged": outcomes[n][0]["diverged"], "selected": n == chosen} for n in plain_names]
    results["gradients"]["layers"] = list(range(1, len(main_path_convs(resnet)) + 1))

    # Extra analysis for the ResNet: per-class accuracy, confusion matrix, sample images for the app
    pred = np.argmax(resnet.predict(data.normalize(x_test), batch_size=256, verbose=0), axis=1)
    confusion = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test, pred):
        confusion[t, p] += 1
    results["resnet_confusion"] = confusion.tolist()
    results["resnet_per_class"] = {c: float(confusion[i, i] / max(confusion[i].sum(), 1))
                                   for i, c in enumerate(data.CLASS_NAMES)}
    sample_idx = np.concatenate([np.where(y_test == c)[0][:3] for c in range(10)])  # first 3 per class, not hand-picked
    np.savez_compressed(out / "samples.npz", images=x_test[sample_idx], labels=y_test[sample_idx])

    save_weights_npz(resnet, out / "resnet_weights.npz")
    (out / "results.json").write_text(json.dumps(results))
    print(f"\nSaved resnet_weights.npz, samples.npz and results.json to {out}/")
    for name, info in results["models"].items():
        print(f"  {name:7s} test accuracy {info['test_accuracy']:.4f}" + ("  (diverged)" if info["diverged"] else ""))
    print("  plain runs:", ", ".join(f"lr {r['peak_lr']}: val {r['final_val_accuracy']:.3f}" + (" *" if r["selected"] else "")
                                     for r in results["plain_runs"]))


if __name__ == "__main__":
    main()
