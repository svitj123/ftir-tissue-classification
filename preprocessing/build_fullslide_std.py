"""Kot build_fullslide.py, a z DODANO per-kanalno standardizacijo (StandardScaler)
PRED PCA fit-om in transformom.

ZAKAJ
-----
Prejsnja meritev: PCA(16) na anotiranih pikslih brez standardizacije zajame
97.53-97.70% variance, clanek za SD porocá 90.03%. Standardizacija (z-score
na 813 kanalih, fit SAMO na tkivu ucne rezine, isti scaler nato uporabljen
na testu) je edina doslej delno-potrjena razlaga vrzeli -- v zgodnejsem
testu na anotiranih pikslih je premaknila varianco 97.68%->93.80% (priblizno
razpolovila vrzel do clanka).

Ta skripta preveri isto na CELEM tkivu (ne le anotiranih pikslih, kot
fit_pca_tissue.py) in -- pomembno -- dejansko zgradi train/test_pca16.npy
v standardiziranem prostoru, da je uporabno za CNN trening (--fullslide-dir).

Metodologija: StandardScaler fit SAMO na (vzorcenih) tkivnih pikslih UCNE
rezine, enako kot PCA. Isti scaler+PCA nato transform obeh rezin. Testne se
ne dotaknemo pri fit-u nobenega koraka.

Izhodna mapa: FTIR-data/fullslide_std/
"""
import glob
import os
import time

import h5py
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Image.MAX_IMAGE_PIXELS = None

TRAIN_DIR = "FTIR-data/train_preprocessed"
TEST_DIR = "FTIR-data/test_preprocessed_full"
OUT_DIR = "FTIR-data/fullslide_std"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_H, TRAIN_W = 3400, 6800
TEST_H, TEST_W = 2800, 6800
N_PCA = 16
SEED = 42

TRAIN_EXCLUDE = {"21", "22", "23"}

MASK_DIRS = {
    "train": "/home/sjesenk/mayerich2-učni/supervised-class",
    "test": "/home/sjesenk/mayerich-testni/supervised-class",
}
CLASS_ORDER = ["coll", "epith", "fibro", "lymph", "myo", "necrosis", "blood"]
ARTICLE_CLASSES = {"blood", "coll", "epith", "myo", "necrosis"}


def nal_mask(p):
    m = np.asarray(Image.open(p))
    return (m.any(-1) if m.ndim == 3 else m) > 0


def crop_list(d, exclude=()):
    poti = sorted(glob.glob(os.path.join(d, "*_crop_*.hdf5")))
    out = []
    for p in poti:
        ime = os.path.basename(p).split("_")[-1].replace(".hdf5", "")
        if ime in exclude:
            continue
        out.append(p)
    return out


def build_labels(H, W, mask_dir, oznaka):
    print(f"\n  [{oznaka}] gradim karte oznak iz PNG mask ({mask_dir})")
    labels7 = np.full((H, W), -1, dtype=np.int8)
    for idx, kratko in enumerate(CLASS_ORDER):
        p = os.path.join(mask_dir, f"class_{kratko}.png")
        m = nal_mask(p)
        assert m.shape == (H, W), f"{p}: oblika {m.shape} != ({H},{W})"
        labels7[m] = idx
    labels5 = labels7.copy()
    labels5[labels7 == CLASS_ORDER.index("fibro")] = -1
    labels5[labels7 == CLASS_ORDER.index("lymph")] = -1
    return labels7, labels5


def sample_tissue_spectra(train_crops, row_stride=8, max_spectra=400_000, seed=SEED):
    print(f"\n  Vzorcim tkivne piksle iz {len(train_crops)} train crop-ov "
          f"(vsaka {row_stride}. vrstica)")
    rng = np.random.RandomState(seed)
    zbrano = []
    t0 = time.time()
    for p in train_crops:
        with h5py.File(p, "r") as f:
            tissue = np.array(f["tissue_mask"])
            vrstice = np.arange(0, tissue.shape[0], row_stride)
            data = f["data"][vrstice, :, :]
        t_sub = tissue[vrstice, :]
        spektri = data[t_sub]
        if len(spektri):
            zbrano.append(spektri.astype(np.float32, copy=False))
        del data, spektri
    X = np.concatenate(zbrano, axis=0)
    del zbrano
    n_bad = int((~np.isfinite(X)).any(axis=1).sum())
    if n_bad:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"    {n_bad} spektrov z NaN/Inf -- sanitizirano")
    if len(X) > max_spectra:
        idx = rng.choice(len(X), max_spectra, replace=False)
        X = X[idx]
    print(f"    {len(X):,} spektrov ({time.time()-t0:.0f}s branja)")
    return X


def fit_scaler_and_pca(X, seed=SEED):
    print(f"\n  Fitam StandardScaler na {len(X):,} x {X.shape[1]} tkivnih spektrih...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print(f"  Racunam PCA({N_PCA}) na standardiziranih spektrih...")
    t1 = time.time()
    pca = PCA(n_components=N_PCA, random_state=seed)
    pca.fit(Xs)
    var = pca.explained_variance_ratio_.sum() * 100
    print(f"    koncano v {time.time()-t1:.0f}s  -- pojasnjena varianca: {var:.2f}%")
    print(f"    (primerjava brez standardizacije: 97.70% na istem tkivu; clanek: 90.03%)")
    return scaler, pca


def transform_slide(crops, scaler, pca, H, W, oznaka):
    print(f"\n  [{oznaka}] transformiram {len(crops)} crop-ov (standardize+PCA) -> {H}x{W}x{N_PCA}")
    out = np.zeros((H, W, N_PCA), dtype=np.float32)
    zajeto = np.zeros((H, W), dtype=bool)
    for p in crops:
        with h5py.File(p, "r") as f:
            a = dict(f.attrs)
            data = np.array(f["data"])
        r0, c0 = int(a["row_start"]), int(a["col_start"])
        h, w, _ = data.shape
        r1, c1 = min(r0 + h, H), min(c0 + w, W)
        flat = data[:r1 - r0, :c1 - c0].reshape(-1, data.shape[-1])
        n_bad = int((~np.isfinite(flat)).any(axis=1).sum())
        if n_bad:
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        flat_s = scaler.transform(flat)
        proj = pca.transform(flat_s).reshape(r1 - r0, c1 - c0, N_PCA).astype(np.float32)
        out[r0:r1, c0:c1] = proj
        zajeto[r0:r1, c0:c1] = True
        del data, flat, flat_s, proj
    print(f"    pokrito: {int(zajeto.sum()):,} / {H*W:,} px")
    return out


def main():
    t_start = time.time()

    train_crops = crop_list(TRAIN_DIR, exclude=TRAIN_EXCLUDE)
    test_crops = crop_list(TEST_DIR, exclude=())
    print(f"TRAIN crop-ov: {len(train_crops)}   TEST crop-ov: {len(test_crops)}")

    scaler_path = os.path.join(OUT_DIR, "scaler_pca_fullslide16.npz")
    if os.path.exists(scaler_path):
        print(f"\n  ponovno uporabljam obstojeci {scaler_path}")
        z = np.load(scaler_path)
        scaler = StandardScaler()
        scaler.mean_ = z["scaler_mean"]; scaler.scale_ = z["scaler_scale"]
        scaler.var_ = z["scaler_scale"] ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
        pca = PCA(n_components=N_PCA, random_state=SEED)
        pca.components_ = z["components"]; pca.mean_ = z["pca_mean"]
        pca.explained_variance_ = z["explained_variance"]
        pca.explained_variance_ratio_ = z["explained_variance_ratio"]
        pca.n_features_in_ = pca.mean_.shape[0]
        print(f"    pojasnjena varianca: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    else:
        X = sample_tissue_spectra(train_crops)
        scaler, pca = fit_scaler_and_pca(X)
        del X
        np.savez_compressed(
            scaler_path,
            scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
            components=pca.components_, pca_mean=pca.mean_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )

    train_path = os.path.join(OUT_DIR, "train_pca16.npy")
    if os.path.exists(train_path):
        print(f"\n  [TRAIN] {train_path} ze obstaja, preskakujem transform")
    else:
        train_pca = transform_slide(train_crops, scaler, pca, TRAIN_H, TRAIN_W, "TRAIN")
        np.save(train_path, train_pca)
        del train_pca

    test_path = os.path.join(OUT_DIR, "test_pca16.npy")
    if os.path.exists(test_path):
        print(f"\n  [TEST] {test_path} ze obstaja, preskakujem transform")
    else:
        test_pca = transform_slide(test_crops, scaler, pca, TEST_H, TEST_W, "TEST")
        np.save(test_path, test_pca)
        del test_pca

    # oznake so identicne originalnemu build_fullslide.py -- kopiramo iz obstojece mape
    src = "FTIR-data/fullslide"
    for fn in ("train_labels7.npy", "train_labels5.npy", "test_labels7.npy", "test_labels5.npy"):
        src_p, dst_p = os.path.join(src, fn), os.path.join(OUT_DIR, fn)
        if os.path.exists(src_p) and not os.path.exists(dst_p):
            import shutil
            shutil.copy(src_p, dst_p)
            print(f"  kopirano: {fn}")
        elif not os.path.exists(src_p):
            train_l7, train_l5 = build_labels(TRAIN_H, TRAIN_W, MASK_DIRS["train"], "TRAIN")
            np.save(os.path.join(OUT_DIR, "train_labels7.npy"), train_l7)
            np.save(os.path.join(OUT_DIR, "train_labels5.npy"), train_l5)
            test_l7, test_l5 = build_labels(TEST_H, TEST_W, MASK_DIRS["test"], "TEST")
            np.save(os.path.join(OUT_DIR, "test_labels7.npy"), test_l7)
            np.save(os.path.join(OUT_DIR, "test_labels5.npy"), test_l5)
            break

    print(f"\nSKUPAJ CAS: {time.time()-t_start:.0f}s")
    print(f"Shranjeno v {OUT_DIR}/")


if __name__ == "__main__":
    main()
