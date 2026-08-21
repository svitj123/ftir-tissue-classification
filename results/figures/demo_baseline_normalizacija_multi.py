"""
Demo: rubber-band baseline correction + Amide I normalizacija na VEC
resnicnih pikslih (po enem iz vsakega od 5 clanku-primerljivih razredov:
coll, epith, myo, necrosis, blood) iz train rezine. Isti algoritem kot
dejanski pipeline (create_train_new_crops.py: rubberband_single,
amide_i_normalize), samo na peščici pikslov namesto celi rezini.

Poganjaj na streznikou (o1/o2.biolab.si), kjer zivijo surovi ENVI podatki:
  python3 demo_baseline_normalizacija_multi.py
Izris se shrani v demo_baseline_normalizacija_multi.png
"""

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spectral.io import envi

DATA_DIR = Path("/home/sjesenk/mayerich2-učni")
HEADER_FILE = DATA_DIR / "br1003-br2085b.hdr"
# train_crop_08 vsebuje vseh 7 native razredov (preverjeno) -- eno crop
# datoteko potrebujemo samo za NAJTI reprezentativne koordinate, surove
# spektre beremo neposredno iz ENVI, ne iz te (ze predobdelane) datoteke.
CROP_FILE = Path("/home/sjesenk/diploma/FTIR-data/train_preprocessed/train_crop_08.hdf5")
SPECTRAL_STEP = 2
AMIDE_I_TARGET_WN = 1650.0
EPS = 1e-6

# native_idx -> (ime, barva) -- iste barve kot povsod drugod v projektu
CLASSES = {
    0: ("coll", "#7F77DD"),
    1: ("epith", "#1D9E75"),
    4: ("myo", "#D85A30"),
    5: ("necrosis", "#D4537E"),
    6: ("blood", "#E24B4A"),
}


def rubberband_single(spectrum: np.ndarray):
    n = len(spectrum)
    x = np.arange(n, dtype=np.float64)
    y = spectrum.astype(np.float64, copy=False)
    lower = []
    for i in range(n):
        point = (x[i], y[i])
        while len(lower) >= 2:
            origin, previous = lower[-2], lower[-1]
            cross = ((previous[0] - origin[0]) * (point[1] - origin[1])
                     - (previous[1] - origin[1]) * (point[0] - origin[0]))
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(point)
    lower_x = np.asarray([p[0] for p in lower], dtype=np.float64)
    lower_y = np.asarray([p[1] for p in lower], dtype=np.float64)
    baseline = np.interp(x, lower_x, lower_y)
    return (y - baseline).astype(np.float32), baseline.astype(np.float32)


def amide_i_normalize(spectrum: np.ndarray, amide_index: int, eps=EPS):
    amide_value = float(spectrum[amide_index])
    if not np.isfinite(amide_value) or amide_value <= eps:
        amide_value = eps
    return (spectrum / amide_value).astype(np.float32)


def find_class_pixels():
    """Za vsak ciljni razred poisce eno GLOBALNO (row, col) koordinato
    znotraj CROP_FILE, ki ima to oznako."""
    with h5py.File(CROP_FILE, "r") as f:
        classes_blood = np.array(f["classes_blood"])
        row_start = int(f.attrs["row_start"])
        col_start = int(f.attrs["col_start"])
    out = {}
    for native_idx, (name, color) in CLASSES.items():
        rows, cols = np.where(classes_blood == native_idx)
        if len(rows) == 0:
            print(f"  OPOZORILO: razred {name} ni najden v {CROP_FILE.name}, preskakujem")
            continue
        mid = len(rows) // 2
        out[native_idx] = (row_start + int(rows[mid]), col_start + int(cols[mid]))
    return out


def main():
    print(f"Iscem po en piksel na razred v {CROP_FILE.name} ...")
    pixels = find_class_pixels()
    for native_idx, (r, c) in pixels.items():
        name = CLASSES[native_idx][0]
        print(f"  {name:<10} -> globalna (r={r}, c={c})")

    print(f"Odpiram surove ENVI podatke: {HEADER_FILE}")
    image = envi.open(str(HEADER_FILE))
    wavelengths_full = np.asarray(
        [float(v) for v in image.metadata["wavelength"]], dtype=np.float32)
    wavelengths = wavelengths_full[::SPECTRAL_STEP]
    amide_index = int(np.abs(wavelengths - AMIDE_I_TARGET_WN).argmin())
    print(f"  Amide I: {wavelengths[amide_index]:.1f} cm^-1 (idx={amide_index})")
    memmap = image.open_memmap(interleave="bip")

    results = {}
    for native_idx, (r, c) in pixels.items():
        name, color = CLASSES[native_idx]
        raw = np.asarray(memmap[r, c, ::SPECTRAL_STEP], dtype=np.float32)
        corrected, baseline = rubberband_single(raw)
        normalized = amide_i_normalize(corrected, amide_index)
        results[name] = dict(color=color, raw=raw, baseline=baseline,
                             corrected=corrected, normalized=normalized)

    print("Izracunano za vse razrede. Rise graf ...")
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    for name, d in results.items():
        axes[0].plot(wavelengths, d["raw"], color=d["color"], linewidth=1.2, label=name)
    axes[0].set_title("1. Surovi spektri (5 razredov, po en piksel)")
    axes[0].set_ylabel("absorbanca (surovo)")
    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    axes[0].invert_xaxis()

    for name, d in results.items():
        axes[1].plot(wavelengths, d["corrected"], color=d["color"], linewidth=1.2, label=name)
    axes[1].axhline(0, color="#898781", linewidth=0.8)
    axes[1].axvline(wavelengths[amide_index], color="black", linewidth=0.8,
                    linestyle=":", label=f"Amide I ({wavelengths[amide_index]:.0f} cm$^{{-1}}$)")
    axes[1].set_title("2. Po rubber-band odstranitvi osnovne linije (vsak spekter svoja osnovnica)")
    axes[1].set_ylabel("absorbanca (popravljeno)")
    axes[1].legend(loc="upper right", fontsize=9, ncol=2)

    for name, d in results.items():
        axes[2].plot(wavelengths, d["normalized"], color=d["color"], linewidth=1.2, label=name)
    axes[2].axhline(1.0, color="#898781", linewidth=0.8, linestyle=":")
    axes[2].axvline(wavelengths[amide_index], color="black", linewidth=0.8, linestyle=":")
    axes[2].set_title("3. Po normalizaciji na Amide I (vsak spekter deljen z LASTNIM Amide I vrhom)")
    axes[2].set_ylabel("absorbanca (normalizirano)")
    axes[2].set_xlabel("valovno stevilo (cm$^{-1}$)")
    axes[2].legend(loc="upper right", fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig("demo_baseline_normalizacija_multi.png", dpi=150)
    fig.savefig("demo_baseline_normalizacija_multi.pdf")  # vektorska, za diplomo
    print("Shranjeno: demo_baseline_normalizacija_multi.png (+.pdf)")


if __name__ == "__main__":
    main()
