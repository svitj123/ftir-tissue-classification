"""
Primerjalni graf OA (vodoravne skupine stolpcev), po vzoru skice, ki jo je
predlagal uporabnik -- clanek proti nasi replikaciji, za tri modele.

Isti podatki kot oa_primerjava.py (10-seed potrjeni rezultati), samo
drugacna (vodoravna) postavitev in stevilke izpisane neposredno ob stolpcih.

Poganjaj lokalno:
  venv/bin/python3 results/figures/oa_primerjava_horizontalna.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (ime, nas OA, nas std, clanek OA, clanek std ali None)
MODELI = [
    ("RBF SVM", 56.55, 0.28, 56.41, 0.27),
    ("Spektralni CNN", 69.98, 3.34, 62.52, None),
    ("Prostorsko-spektralni CNN", 81.56, 3.18, 79.18, None),
]

BARVA_CLANEK = "#9c9488"
BARVA_NASE = "#2f6f62"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
})

fig, ax = plt.subplots(figsize=(7.6, 4.0))

n = len(MODELI)
# clanek NAD nasim rezultatom v vsaki skupini (dosledno s krivuljo/tabelami)
y_clanek = np.arange(n) * 2 - 0.5
y_nase = np.arange(n) * 2 + 0.5

clanek_vals = [m[3] for m in MODELI]
clanek_err = [m[4] if m[4] is not None else 0 for m in MODELI]
nase_vals = [m[1] for m in MODELI]
nase_err = [m[2] for m in MODELI]
imena = [m[0] for m in MODELI]

ax.barh(y_clanek, clanek_vals, height=0.85, xerr=clanek_err, capsize=3,
        color=BARVA_CLANEK, zorder=3)
ax.barh(y_nase, nase_vals, height=0.85, xerr=nase_err, capsize=3,
        color=BARVA_NASE, zorder=3)

for y, val in zip(y_clanek, clanek_vals):
    ax.text(val + 1.3, y, f"{val:.2f}", va="center", fontsize=13, color=BARVA_CLANEK)
for y, val, err in zip(y_nase, nase_vals, nase_err):
    ax.text(val + err + 1.3, y, f"{val:.2f}", va="center", fontsize=13,
            color=BARVA_NASE, fontweight="bold")

# oznaka, kateri stolpec je kateri, neposredno v prvi (zgornji) skupini
# stolpcev, namesto locene legende -- manjsa pisava, da ne prekrije %
ax.text(2, y_clanek[0], "Berisha in sod. (2019)", va="center", ha="left",
        fontsize=9.5, color="white", fontweight="bold")
ax.text(2, y_nase[0], "naša replikacija", va="center", ha="left",
        fontsize=9.5, color="white", fontweight="bold")

ax.set_yticks(np.arange(n) * 2)
ax.set_yticklabels(imena, fontsize=14)
ax.invert_yaxis()  # SVM na vrhu, kot v skici
ax.set_xlabel("Skupna klasifikacijska točnost (OA, %)")
ax.set_xlim(0, 95)
ax.xaxis.grid(True, linestyle=":", linewidth=0.6, color="#c9c2b4", zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(left=False)

fig.tight_layout()
fig.savefig("results/figures/oa_primerjava_horizontalna.pdf")
fig.savefig("results/figures/oa_primerjava_horizontalna.png", dpi=200)
print("Shranjeno: results/figures/oa_primerjava_horizontalna.pdf (+.png za predogled)")
