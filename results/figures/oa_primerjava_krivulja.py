"""
Primerjalni graf OA kot krivulja -- clanek proti nasi replikaciji, cez tri
modele narascajoce kompleksnosti (SVM -> spektralni CNN -> prostorsko-
-spektralni CNN). Preprost, standarden matplotlib slog (brez okrasja).

Poganjaj lokalno:
  venv/bin/python3 results/figures/oa_primerjava_krivulja.py
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

plt.rcParams.update({"font.family": "serif", "font.size": 15})

fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(MODELI))
clanek_vals = [m[3] for m in MODELI]
clanek_err = [m[4] if m[4] is not None else 0 for m in MODELI]
nase_vals = [m[1] for m in MODELI]
nase_err = [m[2] for m in MODELI]
imena = [m[0] for m in MODELI]

ax.errorbar(x, clanek_vals, yerr=clanek_err, marker="s", markersize=8,
           linewidth=2, capsize=4, label="Berisha in sod. (2019)")
ax.errorbar(x, nase_vals, yerr=nase_err, marker="o", markersize=8,
           linewidth=2, capsize=4, label="naša replikacija")

ax.set_xticks(x)
ax.set_xticklabels(imena)
ax.set_ylabel("Skupna klasifikacijska točnost, OA (%)")
ax.set_ylim(45, 90)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.legend(loc="upper left", fontsize=14)

fig.tight_layout()
fig.savefig("results/figures/oa_primerjava_krivulja.pdf")
fig.savefig("results/figures/oa_primerjava_krivulja.png", dpi=200)
print("Shranjeno: results/figures/oa_primerjava_krivulja.pdf (+.png za predogled)")
