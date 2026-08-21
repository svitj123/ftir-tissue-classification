"""
Klasifikacijska mapa testne rezine: Amide I referenca | ground truth |
napoved SVM | napoved spektralnega CNN | napoved prostorsko-spektralnega
(flagship) CNN -- po vzoru uporabnikove originalne zahteve (5 panelov).

H&E slika (brc961-he.ndpi) obstaja, a zahteva registracijo s FTIR
koordinatnim sistemom, ki je izven obsega tega koraka -- namesto nje
uporabimo intenziteto pri Amide I pasu (1650 cm^-1), izracunano iz istih
surovih ENVI podatkov kot ostali predprocesirni koraki v projektu.

Poganjaj NA STREZNIKU (rabi surove ENVI podatke + velike .npz datoteke):
  python3 results/figures/klasifikacijska_mapa.py
Izhod: klasifikacijska_mapa.pdf (+.png za predogled)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from spectral.io import envi

FULLSLIDE_DIR = Path("FTIR-data/fullslide")
CNN_NPZ = Path("modelC_fullslide_faithful_v20_gc05_seed42.npz")
SPECTRAL_NPZ = Path("modelB_spectral_cnn_faithful_seed42.npz")
SVM_NPZ = Path("modelA_svm_rbf_faithful_flagship.npz")
TEST_HEADER = Path("/home/sjesenk/mayerich-testni/brc961-br1001.hdr")
SPECTRAL_STEP = 2
AMIDE_I_TARGET_WN = 1650.0

# ista paleta, ki jo uporabljamo skozi cel projekt
RAZRED_BARVE = {
    "coll": "#7F77DD",
    "epith": "#1D9E75",
    "myo": "#D85A30",
    "necrosis": "#D4537E",
    "blood": "#E24B4A",
}
RAZRED_IME_SL = {
    "coll": "kolagen",
    "epith": "epitelij",
    "myo": "miofibroblasti",
    "necrosis": "nekroza",
    "blood": "kri",
}


def nalozi_amide_referenco():
    print(f"Odpiram surove ENVI podatke: {TEST_HEADER}")
    image = envi.open(str(TEST_HEADER))
    wavelengths_full = np.asarray([float(v) for v in image.metadata["wavelength"]], dtype=np.float32)
    wavelengths = wavelengths_full[::SPECTRAL_STEP]
    amide_index = int(np.abs(wavelengths - AMIDE_I_TARGET_WN).argmin())
    print(f"  Amide I: {wavelengths[amide_index]:.1f} cm^-1 (idx={amide_index})")
    memmap = image.open_memmap(interleave="bip")
    amide_band = np.asarray(memmap[:, :, amide_index * SPECTRAL_STEP], dtype=np.float32)
    return amide_band


def zgradi_gosto_mapo(probs, tissue_mask):
    """Za modele s (H,W,C) gosto napovedjo (prostorski CNN)."""
    H, W = tissue_mask.shape
    out = np.full((H, W), -1, dtype=np.int8)
    pred = np.argmax(probs, axis=-1).astype(np.int8)
    out[tissue_mask] = pred[tissue_mask]
    return out


def zgradi_redko_mapo(probs, coords, shape):
    """Za modele brez prostorskega konteksta (spektralni CNN) -- (N,C)
    verjetnosti samo na testnih koordinatah, ostalo ostane -1 (brez oznake)."""
    out = np.full(shape, -1, dtype=np.int8)
    pred = np.argmax(probs, axis=-1).astype(np.int8)
    out[coords[:, 0], coords[:, 1]] = pred
    return out


def main():
    print("=== Nalaganje ===")
    tissue = np.load(FULLSLIDE_DIR / "test_tissue.npy").astype(bool)
    print(f"  tissue mask: {tissue.shape}, {tissue.sum():,} tkivnih pikslov")

    d_cnn = np.load(CNN_NPZ)
    d_spec = np.load(SPECTRAL_NPZ)
    d_svm = np.load(SVM_NPZ)
    remap_names = [n for n in d_cnn["remap_names"]]
    num_classes = len(remap_names)
    print(f"  razredi: {remap_names}")

    # ground truth mapa (samo anotirani piksli, testni koordinati so skupni vsem modelom)
    coords = d_cnn["coords_test"]
    y_test = d_cnn["y_test"]
    gt_map = np.full(tissue.shape, -1, dtype=np.int8)
    gt_map[coords[:, 0], coords[:, 1]] = y_test

    # MT (mentorjev komentar): brez razloga je bil samo prostorski CNN
    # prikazan na celotnem tkivnem obmocju, ostali le na testnih anotiranih
    # slikovnih elementih -- za pravicno primerjavo vse tri omejimo enako.
    cnn_probs_test = d_cnn["probs"][coords[:, 0], coords[:, 1], :]
    cnn_map = zgradi_redko_mapo(cnn_probs_test, coords, tissue.shape)
    spec_map = zgradi_redko_mapo(d_spec["probs"], d_spec["coords_test"], tissue.shape)
    svm_map = zgradi_redko_mapo(d_svm["probs"], d_svm["coords_test"], tissue.shape)

    amide = nalozi_amide_referenco()

    barve = [RAZRED_BARVE[n] for n in remap_names]
    cmap = ListedColormap(["#e9e5da"] + barve)  # -1 -> svetlo siva (brez tkiva/oznake)

    # OPOMBA: koncna slika se v .tex skrci na \textwidth (~6.3in). Panelov je
    # 5, vsak pa je sam po sebi zelo sirok posnetek (2800x6800), zato jih NE
    # damo v eno vrsto (bi bili trakovi, komaj citljivi) -- namesto tega 2
    # stolpca x 3 vrstice, kar vsak panel poveca za ~2.5x.
    NASLOV_FS = 30
    LEGENDA_FS = 24

    def prikazi(ax, data, title):
        ax.imshow(data + 1, cmap=cmap, vmin=0, vmax=num_classes, interpolation="nearest")
        ax.set_title(title, fontsize=NASLOV_FS, pad=10)
        ax.axis("off")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5))

    axes[0, 0].imshow(amide, cmap="gray")
    axes[0, 0].set_title("Amide I intenziteta (1650 cm$^{-1}$)", fontsize=NASLOV_FS, pad=10)
    axes[0, 0].axis("off")

    # vsi trije modeli prikazani samo na testnih anotiranih slikovnih
    # elementih (glej opombo zgoraj) -- enotno, brez izjeme za CNN
    prikazi(axes[0, 1], gt_map, "Prava oznaka")
    prikazi(axes[1, 0], svm_map, "Napoved: SVM")
    prikazi(axes[1, 1], spec_map, "Napoved: spektralni CNN")
    prikazi(axes[2, 0], cnn_map, "Napoved: prostorsko-spektralni CNN")
    axes[2, 1].axis("off")

    # skupna legenda v prazni celici spodaj desno
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=barve[i], label=RAZRED_IME_SL.get(remap_names[i], remap_names[i]))
               for i in range(num_classes)]
    axes[2, 1].legend(handles=handles, loc="center", frameon=False, ncol=1,
                       fontsize=LEGENDA_FS, handlelength=1.6, handleheight=1.6,
                       labelspacing=1.1)

    fig.tight_layout()
    fig.savefig("klasifikacijska_mapa.pdf")
    fig.savefig("klasifikacijska_mapa.png", dpi=150)
    print("Shranjeno: klasifikacijska_mapa.pdf (+.png za predogled)")


if __name__ == "__main__":
    main()
