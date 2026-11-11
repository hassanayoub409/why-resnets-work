"""Draw the block diagrams used in the app:  python make_diagrams.py  ->  assets/blocks.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

GREEN, AMBER, INK, GREY = "#4F7D32", "#F5B52E", "#222222", "#666666"


def box(ax, x, y, text, w=2.3, h=0.62, face="#FFFFFF", edge=INK, size=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))


def main_path(ax, x, add):
    ys = [7.2, 5.9, 4.6]
    labels = ["Conv 1×1 · BN · ReLU", "Conv 3×3 · BN · ReLU", "Conv 1×1 · BN"]
    ax.text(x, 8.75, "input", ha="center", fontsize=11, color=INK)
    arrow(ax, x, 8.55, x, ys[0] + 0.33)
    for y, label in zip(ys, labels):
        box(ax, x, y, label)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, x, a - 0.33, x, b + 0.33)
    if add:
        ax.add_patch(Circle((x, 3.5), 0.3, facecolor="#FFFFFF", edgecolor=GREEN, linewidth=2.4))
        ax.text(x, 3.5, "+", ha="center", va="center", fontsize=17, color=GREEN, fontweight="bold")
        arrow(ax, x, ys[-1] - 0.33, x, 3.8)
        arrow(ax, x, 3.2, x, 2.55)
    else:
        arrow(ax, x, ys[-1] - 0.33, x, 2.55)
    box(ax, x, 2.25, "ReLU", w=1.2, h=0.55)
    arrow(ax, x, 1.97, x, 1.35)
    ax.text(x, 1.05, "output", ha="center", fontsize=11, color=INK)


def skip_path(ax, x, x_side, with_conv):
    ax.plot([x, x_side], [8.2, 8.2], color=GREEN, lw=3, solid_capstyle="round")
    if with_conv:
        ax.plot([x_side, x_side], [8.2, 6.65], color=GREEN, lw=3)
        box(ax, x_side, 6.2, "Conv 1×1 · BN", w=1.9, face="#FFFFFF", edge=GREEN, size=9.5)
        ax.plot([x_side, x_side], [5.85, 3.5], color=GREEN, lw=3)
    else:
        ax.plot([x_side, x_side], [8.2, 3.5], color=GREEN, lw=3)
    arrow(ax, x_side, 3.5, x + 0.33, 3.5, color=GREEN, lw=3)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.3), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    specs = [("Plain block", None, "No shortcut: the signal must\npass through every layer."),
             ("Identity block", False, "Skip connection: the input is\nadded back to the output."),
             ("Convolutional block", True, "Same idea, but a 1×1 convolution on\nthe shortcut makes the shapes match.")]
    for ax, (title, with_conv, note) in zip(axes, specs):
        ax.set_xlim(0, 4.9 if with_conv else 4.2)
        ax.set_ylim(0.2, 9.4)
        ax.axis("off")
        x = 1.5
        main_path(ax, x, add=with_conv is not None)
        if with_conv is not None:
            skip_path(ax, x, 3.5 if with_conv is False else 3.6, with_conv)
        ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=6)
        ax.text(x + 0.6, 0.35, note, ha="center", va="top", fontsize=9.5, color=GREY)
    fig.tight_layout()
    fig.savefig("assets/blocks.png", dpi=110, facecolor="white")
    print("saved assets/blocks.png")


if __name__ == "__main__":
    main()
