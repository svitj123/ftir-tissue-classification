"""
Matrika zmede kot LaTeX tabela (namesto zavrnjenega 3D grafa) -- vsaka
celica prikaze SVM in CNN skupaj, po vzoru mentorjevega predloga.

Poganjaj lokalno:
  venv/bin/python3 results/figures/matrika_zmede_tabela.py
Izhod: natisne LaTeX tabularx blok na stdout.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).parent
CNN_NPZ = HERE / "modelC_fullslide_faithful_v20_gc05_seed42.npz"
SVM_NPZ = HERE / "modelA_svm_rbf_faithful_flagship.npz"

IME_SL = {"coll": "kolagen", "epith": "epitelij", "myo": "miofibroblasti",
          "necrosis": "nekroza", "blood": "kri"}


def pripravi(npz_path):
    d = np.load(npz_path)
    probs = d["probs"]
    y_test = d["y_test"]
    if probs.ndim == 3:
        coords = d["coords_test"]
        probs = probs[coords[:, 0], coords[:, 1], :]
    y_pred = np.argmax(probs, axis=-1)
    names = [str(n) for n in d["remap_names"]]
    return y_test, y_pred, names


def main():
    y_svm, yp_svm, names = pripravi(SVM_NPZ)
    y_cnn, yp_cnn, names_cnn = pripravi(CNN_NPZ)
    assert names == names_cnn
    n = len(names)
    imena_sl = [IME_SL.get(nm, nm) for nm in names]

    cm_svm = confusion_matrix(y_svm, yp_svm, labels=range(n))
    cm_svm_pct = 100 * cm_svm / cm_svm.sum(axis=1, keepdims=True)
    cm_cnn = confusion_matrix(y_cnn, yp_cnn, labels=range(n))
    cm_cnn_pct = 100 * cm_cnn / cm_cnn.sum(axis=1, keepdims=True)

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    col_spec = "@{}l" + "c" * n + "@{}"
    print(r"\begin{tabular}{%s}" % col_spec)
    print(r"\toprule")
    header = "pravi razred $\\backslash$ napoved & " + " & ".join(imena_sl) + r" \\"
    print(header)
    print(r"\midrule")
    def fmt(v):
        return ("%.1f" % v).replace(".", "{,}")

    for i in range(n):
        row = [imena_sl[i]]
        for j in range(n):
            svm_v = cm_svm_pct[i, j]
            cnn_v = cm_cnn_pct[i, j]
            pair = r"$%s$~/~$%s$" % (fmt(svm_v), fmt(cnn_v))
            if i == j:
                cell = r"\textbf{%s}" % pair
            else:
                cell = pair if (svm_v >= 0.05 or cnn_v >= 0.05) else "--"
            row.append(cell)
        print(" & ".join(row) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Matrika zmede za RBF SVM in prostorsko-spektralni CNN skupaj "
          r"(vsaka celica: SVM~/~CNN, \% vzorcev pravega razreda v vrstici, "
          r"normalizirano po vrsticah). Diagonala (poudarjeno) je recall na "
          r"razred; izven diagonale pokaže, kam posamezen model najpogosteje "
          r"zgreši.}")
    print(r"\label{fig:matrika-zmede}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
