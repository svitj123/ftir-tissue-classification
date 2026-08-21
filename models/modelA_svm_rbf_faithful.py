import argparse
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

NATIVE_NAMES = ["coll", "epith", "fibro", "lymph", "myo", "necrosis", "blood"]


def load_labels(path, label_set=5):
    native = np.load(path)
    present = sorted(int(v) for v in np.unique(native) if v != -1)
    if label_set == 5:
        pricakovano = [0, 1, 4, 5, 6]
        if present != pricakovano:
            raise ValueError(
                f"{path}: labels5 pricakuje prisotne native indekse {pricakovano}, "
                f"najdeno {present}")
    remap = -np.ones(7, dtype=np.int64)
    remap_info = []
    for new_idx, native_idx in enumerate(present):
        remap[native_idx] = new_idx
        remap_info.append((native_idx, NATIVE_NAMES[native_idx], new_idx))
    out = np.where(native != -1, remap[np.clip(native, 0, 6)], -1).astype(np.int8)
    return out, remap_info


def oversample_pool(X, y, num_classes, seed, target=10000):
    """Vsak razred nadvzorči/podvzorči na točno `target` vzorcev: pod
    ciljem nadvzorčenje z vračanjem, nad ciljem podvzorčenje brez vračanja."""
    rng = np.random.RandomState(seed)
    idx_parts = []
    for c in range(num_classes):
        cls_idx = np.where(y == c)[0]
        if len(cls_idx) == 0:
            continue
        if len(cls_idx) < target:
            extra = rng.choice(cls_idx, size=target - len(cls_idx), replace=True)
            idx_parts.append(np.concatenate([cls_idx, extra]))
        else:
            idx_parts.append(rng.choice(cls_idx, size=target, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def print_per_class_table(y_true, y_pred, probs, num_classes, remap_info, title):
    ime_po_idx = {new: name for _, name, new in remap_info}
    print(f"\n  {title}")
    print(f"  {'Razred':>14}  {'N':>7}  {'OA':>8}  {'Log-loss':>10}")
    print(f"  {'-'*14}  {'-'*7}  {'-'*8}  {'-'*10}")
    for c in range(num_classes):
        mask = (y_true == c)
        naziv = f"R{c} {ime_po_idx.get(c, '?')}"
        if mask.sum() == 0:
            print(f"  {naziv:>14}  {'-':>7}  {'-':>8}  {'-':>10}")
            continue
        oa_c = accuracy_score(y_true[mask], y_pred[mask])
        ll_c = log_loss(y_true[mask], probs[mask], labels=np.arange(num_classes))
        print(f"  {naziv:>14}  {mask.sum():>7}  {oa_c*100:>7.2f}%  {ll_c:>10.5f}")
    oa_tot = accuracy_score(y_true, y_pred)
    ll_tot = log_loss(y_true, probs, labels=np.arange(num_classes))
    print(f"  {'SKUPAJ':>14}  {len(y_true):>7}  {oa_tot*100:>7.2f}%  {ll_tot:>10.5f}")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h {m}min {s}s"
    if m: return f"{m}min {s}s"
    return f"{s}s"


def main():
    run_start = time.time()
    parser = argparse.ArgumentParser(description="RBF SVM baseline po clanku (Section 2.4/3.1).")
    parser.add_argument("--fullslide-dir", default="FTIR-data/fullslide")
    parser.add_argument("--label-set", type=int, choices=[5, 7], default=5)
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Clanek: 10 neodvisnih ponovitev z novim nakljucnim vzorcenjem.")
    parser.add_argument("--samples-per-class", type=int, default=10000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale_1_16",
                        help="'scale_1_16' = 1/n_features=0.0625 (clankova formula), "
                             "ali poljubno stevilo.")
    parser.add_argument("--no-probability", action="store_true",
                        help="Izklopi Platt scaling (probability=True) -- HITREJE "
                             "(brez interne 5-fold CV), a brez log-loss/AUC, samo OA.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--save-probs", type=str, default=None,
                        help="Ce podano, shrani probs/coords_test/y_test/imena razredov "
                             "iz ZADNJE ponovitve v .npz (isti format kot CNN skripte, "
                             "za ROC/AUC primerjavo -- rabi --n-repeats brez --no-probability).")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Ce podano, shrani natreniran OneVsRestClassifier (SVC) "
                             "iz ZADNJE ponovitve v .joblib.")
    args = parser.parse_args()

    print("\n=== 1. Nalaganje celorezinskih PCA podatkov ===")
    d = args.fullslide_dir
    train_pca = np.load(f"{d}/train_pca16.npy")
    test_pca = np.load(f"{d}/test_pca16.npy")
    train_labels, remap_info = load_labels(f"{d}/train_labels{args.label_set}.npy", args.label_set)
    test_labels, remap_info_test = load_labels(f"{d}/test_labels{args.label_set}.npy", args.label_set)
    assert [r[0] for r in remap_info] == [r[0] for r in remap_info_test], \
        "Train in test labels imata razlicen nabor prisotnih razredov -- prekinjam."
    num_classes = len(remap_info)
    n_pca = train_pca.shape[-1]
    print(f"  train_pca16: {train_pca.shape}   test_pca16: {test_pca.shape}")
    print(f"  NUM_CLASSES = {num_classes} (label-set={args.label_set}) -- OPOMBA: brez "
          f"adipocitov (ni v nasih podatkih), clankov SD ima 6 razredov")

    gamma = 1.0 / n_pca if args.gamma == "scale_1_16" else float(args.gamma)
    print(f"\n  Konfiguracija: SVC(kernel=rbf, C={args.C}, gamma={gamma:.4f}, "
          f"probability={not args.no_probability}) | one-vs-rest | "
          f"{args.samples_per_class:,} vzorcev/razred | {args.n_repeats} ponovitev")

    coords_train = np.argwhere(train_labels != -1)
    y_train_full = train_labels[coords_train[:, 0], coords_train[:, 1]].astype(np.int64)
    X_train_full = train_pca[coords_train[:, 0], coords_train[:, 1]]
    print(f"\n  Train (celotna anotirana rezina): {len(y_train_full):,} pikslov, "
          f"razporeditev: {list(np.bincount(y_train_full, minlength=num_classes))}")

    coords_test = np.argwhere(test_labels != -1)
    y_test = test_labels[coords_test[:, 0], coords_test[:, 1]].astype(np.int64)
    X_test = test_pca[coords_test[:, 0], coords_test[:, 1]]
    print(f"  Test (celotna anotirana rezina, NIKOLI vzorcena): {len(y_test):,} pikslov")

    results = []
    for rep in range(args.n_repeats):
        seed = args.seed_base + rep
        t0 = time.time()
        X_train, y_train = oversample_pool(X_train_full, y_train_full, num_classes,
                                           seed=seed, target=args.samples_per_class)
        print(f"\n=== Ponovitev {rep+1}/{args.n_repeats} (seed={seed}) — "
              f"train: {len(y_train):,} vzorcev ===")

        svc = SVC(kernel="rbf", C=args.C, gamma=gamma,
                 probability=not args.no_probability, random_state=seed)
        clf = OneVsRestClassifier(svc, n_jobs=args.n_jobs)
        clf.fit(X_train, y_train)
        print(f"  Trening koncan v {time.time()-t0:.1f}s")

        y_pred = clf.predict(X_test)
        oa = accuracy_score(y_test, y_pred)
        if not args.no_probability:
            probs = clf.predict_proba(X_test)
            ll = log_loss(y_test, probs, labels=np.arange(num_classes))
        else:
            probs = None
            ll = float("nan")
        print(f"  TEST OA={oa*100:.2f}%  ll={ll:.5f}  ({time.time()-t0:.1f}s skupaj)")

        per_class_oa = []
        for c in range(num_classes):
            mask = y_test == c
            per_class_oa.append(accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() else float("nan"))
        results.append(dict(oa=oa, ll=ll, per_class_oa=per_class_oa))

        if probs is not None:
            print_per_class_table(y_test, y_pred, probs, num_classes, remap_info,
                                  f"Per-class OA/ll (ponovitev {rep+1}):")

        if args.save_probs and rep == args.n_repeats - 1:
            if probs is None:
                print("  OPOZORILO: --save-probs zahteva verjetnosti (brez --no-probability) -- preskocim shranjevanje.")
            else:
                remap_names = np.array([name for _, name, _ in remap_info])
                remap_native_idx = np.array([native for native, _, _ in remap_info])
                np.savez_compressed(args.save_probs, probs=probs.astype(np.float32),
                                    coords_test=coords_test, y_test=y_test,
                                    remap_native_idx=remap_native_idx, remap_names=remap_names)
                print(f"  Shranjeno (za ROC/AUC): {args.save_probs}")

        if args.save_model and rep == args.n_repeats - 1:
            joblib.dump(clf, args.save_model)
            print(f"  Shranjen natreniran model: {args.save_model}")

    print("\n" + "=" * 70)
    print("  POVZETEK -- mean +/- std cez ponovitve")
    print("=" * 70)
    oa_arr = np.array([r["oa"] for r in results]) * 100
    ll_arr = np.array([r["ll"] for r in results])
    print(f"  TEST OA: {oa_arr.mean():.2f} +/- {oa_arr.std(ddof=1):.2f} %")
    if not args.no_probability:
        print(f"  TEST log-loss: {ll_arr.mean():.5f} +/- {ll_arr.std(ddof=1):.5f}")
    print(f"\n  Ref. clanek (Tabela 3, SD, 6 razredov): OA=56.41+/-0.27")
    print(f"  (nasa verzija ima 5 razredov -- brez adipocitov -- primerjava je okvirna)")

    print(f"\n  Per-class OA (mean +/- std):")
    ime_po_idx = {new: name for _, name, new in remap_info}
    per_class_arr = np.array([r["per_class_oa"] for r in results]) * 100
    for c in range(num_classes):
        print(f"    R{c} {ime_po_idx.get(c,'?'):<10} "
              f"{per_class_arr[:,c].mean():>6.2f} +/- {per_class_arr[:,c].std(ddof=1):>5.2f} %")

    print(f"\nSKUPAJ CAS: {format_duration(time.time()-run_start)}")


if __name__ == "__main__":
    main()
