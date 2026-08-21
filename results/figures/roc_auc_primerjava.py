"""
ROC krivulje (one-vs-rest) po razredih -- primerjava spektralnega in
prostorsko-spektralnega (flagship) CNN. Po vzoru clankove Figure 5.

SVM je IZPUSCEN (ne shranjuje verjetnosti na disk, glej plan).

Poganjaj NA STREZNIKU ALI LOKALNO (samo .npz, brez surovih ENVI podatkov):
  python3 results/figures/roc_auc_primerjava.py
Izhod: roc_auc_primerjava.pdf (+.png za predogled)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

CNN_NPZ = "modelC_fullslide_faithful_v20_gc05_seed42.npz"
SPECTRAL_NPZ = "modelB_spectral_cnn_faithful_seed42.npz"

BARVA_CNN = "#2f6f62"
BARVA_SPEKTRALNI = "#a8493c"

IME_SL = {"coll": "kolagen", "epith": "epitelij", "myo": "miofibroblasti",
          "necrosis": "nekroza", "blood": "kri"}


def one_vs_rest_probs_at_test(npz_path):
    d = np.load(npz_path)
    probs = d["probs"]
    y_test = d["y_test"]
    if probs.ndim == 3:
        # gosta (H, W, C) mapa (prostorski CNN) -- izluscimo samo testne koordinate
        coords = d["coords_test"]
        probs = probs[coords[:, 0], coords[:, 1], :]
    # ce je ze (N, C) (spektralni model, ni prostorskega konteksta), pustimo kot je
    return probs, y_test, [str(n) for n in d["remap_names"]]


def main():
    probs_cnn, y_cnn, names = one_vs_rest_probs_at_test(CNN_NPZ)
    probs_spec, y_spec, names_spec = one_vs_rest_probs_at_test(SPECTRAL_NPZ)
    assert names == names_spec, "Razredi se ne ujemajo med modeloma!"
    num_classes = len(names)

    # OPOMBA: koncna slika se v .tex skrci na \textwidth (~6.3in). 5 panelov
    # V ENI VRSTI bi bilo komaj citljivih (faktor skrcitve ~0.24) -- namesto
    # tega 2 stolpca x 3 vrstice (zadnja celica prazna), kar vsak panel
    # priblizno podvoji glede na eno vrsto.
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 17,
    })

    ncols, nrows = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.6 * nrows))
    axes_flat = axes.flatten()

    for c in range(num_classes):
        ax = axes_flat[c]
        y_true_cnn = (y_cnn == c).astype(int)
        y_true_spec = (y_spec == c).astype(int)

        fpr_c, tpr_c, _ = roc_curve(y_true_cnn, probs_cnn[:, c])
        auc_c = auc(fpr_c, tpr_c)
        fpr_s, tpr_s, _ = roc_curve(y_true_spec, probs_spec[:, c])
        auc_s = auc(fpr_s, tpr_s)

        ax.plot(fpr_s, tpr_s, color=BARVA_SPEKTRALNI, linewidth=3.0,
                label=f"spektralni (AUC={auc_s:.3f})")
        ax.plot(fpr_c, tpr_c, color=BARVA_CNN, linewidth=3.2,
                label=f"prostorski (AUC={auc_c:.3f})")
        ax.plot([0, 1], [0, 1], color="#c9c2b4", linewidth=1.4, linestyle=":")

        ime = IME_SL.get(names[c], names[c])
        ax.set_title(ime, pad=10)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right", frameon=False, handlelength=1.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for c in range(num_classes, nrows * ncols):
        axes_flat[c].axis("off")

    fig.tight_layout()
    fig.savefig("roc_auc_primerjava.pdf")
    fig.savefig("roc_auc_primerjava.png", dpi=150)
    print("Shranjeno: roc_auc_primerjava.pdf (+.png za predogled)")


if __name__ == "__main__":
    main()
