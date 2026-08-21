"""
Uvodna slika (BZ zahteva, vrstica ~431 v .tex): (a) reprezentativen izsek
tkiva z oznacenimi razredi, (b) dva spektra razlicnih razredov, da se
nakaze klasifikacijski problem.

Poganjaj NA STREZNIKU (bere train_crop_08.hdf5):
  python3 uvod_problem.py
Izhod: uvod_problem.pdf (+.png za predogled)
"""

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import numpy as np

CROP_FILE = Path("/home/sjesenk/diploma/FTIR-data/train_preprocessed/train_crop_08.hdf5")
# najdeno okno (najboljse razmerje: tkivna gostota + stevilo razlicnih razredov)
R0, C0, SIZE = 260, 520, 180
AMIDE_I_WN = 1650.0

NATIVE_IME = {0: "kolagen", 1: "epitelij", 4: "miofibroblasti", 5: "nekroza", 6: "kri"}
NATIVE_BARVA = {
    0: "#7F77DD", 1: "#1D9E75", 4: "#D85A30", 5: "#D4537E", 6: "#E24B4A",
}

plt.rcParams.update({"font.family": "serif", "font.size": 13})


def main():
    with h5py.File(CROP_FILE, "r") as f:
        wns = np.array(f["wns"])
        classes = np.array(f["classes_blood"][R0:R0 + SIZE, C0:C0 + SIZE])
        amide_idx = int(np.abs(wns - AMIDE_I_WN).argmin())
        amide_img = np.array(f["data"][R0:R0 + SIZE, C0:C0 + SIZE, amide_idx])

        # 2 reprezentativna spektra (razlicna razreda) iz CELEGA izseka,
        # ne samo tega majhnega okna -- izberemo epitelij in nekrozo
        full_classes = np.array(f["classes_blood"])
        spekter_razredi = [1, 5]  # epitelij, nekroza
        spektri = {}
        for c in spekter_razredi:
            rows, cols = np.where(full_classes == c)
            mid = len(rows) // 2
            spektri[c] = np.array(f["data"][rows[mid], cols[mid], :])

    plt.rcParams.update({"font.size": 24})
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(22, 10),
                                          gridspec_kw={"width_ratios": [1, 1.15]})

    # --- (a) tkivo z oznakami ---
    # percentilno rezanje kontrasta namesto surovega min/max -- surov min/max
    # naredi sliko skoraj binarno (crno ozadje/belo tkivo, "cudno" izgleda)
    vmin, vmax = np.percentile(amide_img, [2, 98])
    ax_img.imshow(amide_img, cmap="gray", vmin=vmin, vmax=vmax)
    overlay = np.zeros((SIZE, SIZE, 4))
    for native_idx, barva in NATIVE_BARVA.items():
        mask = classes == native_idx
        if not mask.any():
            continue
        rgb = tuple(int(barva.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        overlay[mask] = (*rgb, 0.55)
    ax_img.imshow(overlay)
    ax_img.axis("off")
    ax_img.text(0.02, 0.98, "(a)", transform=ax_img.transAxes, fontsize=30,
               fontweight="bold", color="white", va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"))

    # oznake z npuscicami za razrede prisotne v tem oknu
    prisotni = [c for c in NATIVE_BARVA if (classes == c).any()]
    label_pozicije = [(0.06, 0.90), (0.62, 0.88), (0.08, 0.15), (0.68, 0.18)]
    for (native_idx, (lx, ly)) in zip(prisotni, label_pozicije):
        rows, cols = np.where(classes == native_idx)
        mid = len(rows) // 2
        ty, tx = rows[mid] / SIZE, cols[mid] / SIZE
        # crna puscica z belim obrobom (path_effects) -- vidna na svetlem IN
        # temnem ozadju, za razliko od prejsnje cisto bele puscice
        ax_img.annotate(
            "", xy=(tx, 1 - ty), xycoords="axes fraction",
            xytext=(lx, ly), textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>", color="black", linewidth=2.6,
                shrinkA=2, shrinkB=6,
                path_effects=[pe.Stroke(linewidth=4.6, foreground="white"), pe.Normal()],
            ))
        ax_img.annotate(
            NATIVE_IME[native_idx], xy=(tx, 1 - ty), xycoords="axes fraction",
            xytext=(lx, ly), textcoords="axes fraction",
            fontsize=22, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=NATIVE_BARVA[native_idx],
                      edgecolor="none", alpha=0.92))

    # --- (b) 2 spektra razlicnih razredov ---
    for c in spekter_razredi:
        ax_spec.plot(wns, spektri[c], color=NATIVE_BARVA[c], linewidth=2.8,
                     label=NATIVE_IME[c])
    ax_spec.invert_xaxis()
    ax_spec.set_xlabel("valovno število (cm$^{-1}$)")
    ax_spec.set_ylabel("absorbanca (predobdelano)")
    ax_spec.text(0.03, 0.97, "(b)", transform=ax_spec.transAxes, fontsize=30,
                fontweight="bold", va="top", ha="left")
    ax_spec.legend(loc="upper right", frameon=False, fontsize=22)
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig("uvod_problem.pdf")
    fig.savefig("uvod_problem.png", dpi=180)
    print("Shranjeno: uvod_problem.pdf (+.png za predogled)")


if __name__ == "__main__":
    main()
