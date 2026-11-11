# 🧱 Why ResNets Work

Build ResNet blocks in TensorFlow/Keras, train a **56-layer ResNet** and the **exact same network without skip connections** on CIFAR-10, and see what skip connections actually change. A Streamlit app lets you classify photos with the trained ResNet (with a Grad-CAM map of where it looked) and compare the two networks.

## What's in the app

1. **Classify a photo:** pick a CIFAR-10 test image or upload your own. The app shows what the network actually sees (32×32 pixels), its answer, and a Grad-CAM map of the regions that drove the answer.
2. **Why skip connections?** Block diagrams, then the comparison: test accuracy and learning curves of the network with and without skip connections, plus the size of the learning signal (gradient) at every layer before and after training. The text on this tab is generated from the measured results, so it reports what happened, not what the textbook says should happen.

## Get it running (3 steps)

The trained weights are not in the repo until you create them (the training takes a GPU; Colab provides one for free).

1. **Train on Google Colab:** open `notebooks/train_on_colab.ipynb` in Colab, switch the runtime to a T4 GPU, and run the cells (about 30 to 90 minutes). It downloads `trained_models.zip`.
2. **Copy the three files** from that zip (`resnet_weights.npz`, `results.json`, `samples.npz`) into the `models/` folder.
3. **Run the app:**

```bash
git clone https://github.com/hassanayoub409/why-resents-work.git
cd why-resents-work
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.9 to 3.12 (TensorFlow 2.16.2 has no 3.13 build). TensorFlow 2.16.2 is pinned because it is the last release with wheels for Intel Macs. Weights trained on Colab's newer TensorFlow load fine (checked: identical predictions on 2.21 and 2.16.2).

To deploy on Streamlit Community Cloud, commit the three files in `models/` (about 3 MB in total) and choose Python 3.12 in the advanced settings.

## Training details

`train.py` (run by the notebook) trains, with identical seeds and settings:
- the **ResNet-56**,
- the **same network without shortcuts** at the same learning rate, and
- the plain network again at a **lower learning rate** (0.02), because a plain network may simply need a gentler one. The plain run with the better *validation* accuracy is the one reported, so the comparison is fair and the test set is never used to choose.

Recipe: 30 epochs, batch 128, SGD with Nesterov momentum, 1 epoch of warm-up then cosine decay, weight decay, random crops and flips. It also measures the gradient size of every convolution before and after training. Quick check without any download: `python train.py --fake-data --epochs 2 --blocks 1 --out /tmp/smoke`.

## Project structure

```
app.py                         # Streamlit app
train.py                       # Trains ResNet and plain networks, measures gradients, saves everything
src/resnet.py                  # identity_block, convolutional_block, build_model(skip=True/False)
src/analysis.py                # Gradient measurement, Grad-CAM, portable weight saving
src/data.py                    # CIFAR-10 loading and augmentation
make_diagrams.py               # Draws assets/blocks.png
notebooks/train_on_colab.ipynb # One-click GPU training
models/                        # Put resnet_weights.npz, results.json, samples.npz here
tests/                         # python -m pytest -q
```

## Limitations

- CIFAR-10 images are only 32×32 and cover 10 classes, so the network is poor on everyday photos with other subjects or busy backgrounds.
- One training run per network: small differences between them are not meaningful.
- The lab trains ResNet-50 on the SIGNS hand-sign data; this project uses the public CIFAR-10 dataset instead and a CIFAR-sized ResNet.

## Data and acknowledgments

- CIFAR-10: A. Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*. Downloaded automatically by Keras. A few test images are included in `samples.npz` for the app's sample picker.
- ResNet: K. He, X. Zhang, S. Ren, J. Sun (2015), *Deep Residual Learning for Image Recognition*.
- The concepts were learned in the *Convolutional Neural Networks* course of the Deep Learning Specialization by DeepLearning.AI on Coursera. The code in this repository was written independently and contains no course materials or datasets. This project is not affiliated with or endorsed by Coursera or DeepLearning.AI.

## License

Code: MIT, see [LICENSE](LICENSE).
