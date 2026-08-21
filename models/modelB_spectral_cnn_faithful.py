"""
Model B -- spektralni CNN, po Berisha in sod. (2019, poglavje 2.5). Uporabi
popolnoma enako arhitekturo (SingleStreamCNN) kot prostorsko-spektralni
model (Model C), a z okolico velikosti 1x1 namesto 17x17 -- napoved torej
temelji izključno na spektru (16 komponent PCA) enega slikovnega elementa,
brez prostorskega konteksta.

Podatki so celorezinski, enaki kot pri Modelu C (glej build_fullslide_std.py):
  FTIR-data/fullslide/{train,test}_pca16.npy   -- 16-kan. PCA projekcije, cela rezina
  FTIR-data/fullslide/{train,test}_labels5.npy -- oznake, 5 skupnih razredov

Ključne odločitve, kjer članek ni eksplicitno naveden:
  - Nadvzorčenje na 100.000 vzorcev na razred, po členku zvestem receptu.
  - Optimizator je privzeto Adam (namesto členku zveste Adadelte), ker
    Adadelta na tem modelu popolnoma kolabira -- članek za spektralni
    model sploh ne poda učnega recepta, zato je Adam legitimna prilagoditev.

Zagon:
  python3 modelB_spectral_cnn_faithful.py --seed 42 \\
      --output modelB_spectral_cnn_faithful_seed42.npz
"""

import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset

RESULTS_FILE = "rezultati_report.txt"
NUM_WORKERS = 0
NUM_CLASSES = 5
WEIGHT_SOFTEN = 1.0  # uporabljen samo pri --balance-strategy weights

NATIVE_NAMES = ["coll", "epith", "fibro", "lymph", "myo", "necrosis", "blood"]


# ---------------------------------------------------------------------------
# Oznake
# ---------------------------------------------------------------------------
def load_labels(path, label_set):
    native = np.load(path)
    present = sorted(int(v) for v in np.unique(native) if v != -1)
    if label_set == 5:
        pricakovano = [0, 1, 4, 5, 6]  # coll, epith, myo, necrosis, blood
        if present != pricakovano:
            raise ValueError(
                f"{path}: labels5 pricakuje prisotne native indekse {pricakovano}, "
                f"najdeno {present} -- preveri, da je datoteka res labels5.npy")
    remap = -np.ones(7, dtype=np.int64)
    remap_info = []
    for new_idx, native_idx in enumerate(present):
        remap[native_idx] = new_idx
        remap_info.append((native_idx, NATIVE_NAMES[native_idx], new_idx))
    out = np.where(native != -1, remap[np.clip(native, 0, 6)], -1).astype(np.int8)
    return out, remap_info


def print_remap(remap_info, label):
    print(f"  [{label}] Remap oznak (native -> strnjen 0..{len(remap_info)-1}):")
    for native_idx, name, new_idx in remap_info:
        print(f"    R{native_idx} {name:<10} -> {new_idx}")


# ---------------------------------------------------------------------------
# Dataset — SAMO spekter (PCA(16) vektor) enega piksla, BREZ okolice
# ---------------------------------------------------------------------------
class SpectralDataset(Dataset):
    """samples: (N,2) int array (r,c) koordinat v PCA sliki."""
    def __init__(self, pca_cube, samples, labels):
        self.pca_cube = pca_cube
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r, c = self.samples[idx]
        spectrum = self.pca_cube[r, c]                       # (n_pca,)
        x = torch.from_numpy(spectrum.copy()).view(-1, 1, 1)  # (n_pca, 1, 1)
        return x, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Arhitektura -- ista kot prostorsko-spektralni CNN (Berisha in sod. Fig. 3, SD = brez BN),
# uporabljena na (n_pca,1,1) vhodu namesto (n_pca,17,17)
# ---------------------------------------------------------------------------
class SingleStreamCNN(nn.Module):
    def __init__(self, n_channels, num_classes, dropout=0.5, use_lrn=False):
        super().__init__()
        lrn = (lambda: nn.LocalResponseNorm(size=5)) if use_lrn else (lambda: nn.Identity())
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.Softplus(), nn.Dropout(dropout),
            lrn(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Softplus(), nn.Dropout(dropout),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Softplus(), nn.Dropout(dropout),
            lrn(),
            nn.MaxPool2d(2, ceil_mode=True),
        )
        with torch.no_grad():
            dummy_out = self.features(torch.zeros(1, n_channels, 1, 1))
        self.flatten_dim = dummy_out.numel()
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_logits(self, x):
        f = self.features(x).reshape(x.size(0), -1)
        f = F.softplus(self.fc1(f))
        f = self.dropout(f)
        return self.fc2(f)

    def forward(self, x):
        return self.log_softmax(self.get_logits(x))


# ---------------------------------------------------------------------------
# Balansiranje razredov
# ---------------------------------------------------------------------------
def compute_class_weights(y, num_classes, soften=WEIGHT_SOFTEN):
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    raw     = len(y) / (num_classes * np.where(counts > 0, counts, 1))
    weights = raw ** soften
    print(f"  Utezi razredov (soften={soften}): {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


def oversample_pool(samples, y, num_classes, seed=42, target=None):
    """Nadvzorčenje po členku zvestem receptu -- fiksen nabor vzorcev,
    zgrajen enkrat, nato le premešan po epohah."""
    rng = np.random.RandomState(seed)
    counts = np.bincount(y, minlength=num_classes)
    target = int(counts.max()) if target is None else int(target)
    print(f"  Nadvzorcenje: {list(counts)} -> {target}/razred")
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
    print(f"  Skupaj po nadvzorcenju: {len(idx):,} (prej {len(y):,})")
    return samples[idx], y[idx]


class BalancedOnTheFlySampler(torch.utils.data.Sampler):
    """Za vsak vzorec: nakljucen razred (uniformno), nato nakljucen original iz
    tega razreda -- SVEZE vsako epoho, brez fiksnega seznama kopij."""
    def __init__(self, y, num_classes, epoch_len=None, seed=42):
        self.class_indices = [np.where(y == c)[0] for c in range(num_classes)]
        self.class_indices = [ci for ci in self.class_indices if len(ci)]
        self.num_classes = len(self.class_indices)
        self.epoch_len = epoch_len if epoch_len is not None else len(y)
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        cls_choice = self.rng.randint(0, self.num_classes, size=self.epoch_len)
        for c in cls_choice:
            ci = self.class_indices[c]
            yield int(ci[self.rng.randint(0, len(ci))])

    def __len__(self):
        return self.epoch_len


def build_criterion(strategy, weights, device):
    if strategy in ("oversample", "oversample_batch"):
        return nn.NLLLoss()
    return nn.NLLLoss(weight=weights.to(device))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda"); print(f"  Naprava: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h {m}min {s}s"
    if m: return f"{m}min {s}s"
    return f"{s}s"


def print_per_class_table(y_true, y_pred, probs, num_classes, remap_info,
                          title="Rezultati po razredih"):
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


def write_results_report(model_name, test_oa, test_ll, output_path, final_epochs,
                          extra_note="", total_duration_str=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dur_part = f"  cas={total_duration_str}" if total_duration_str else ""
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"TEST_OA={test_oa*100:6.2f}%  TEST_ll={test_ll:.5f}  "
        f"final_ep={final_epochs}{dur_part}"
        f"  -> {output_path}\n"
    ]
    if extra_note:
        lines.append(f"{'':>19}  Opomba: {extra_note}\n")
    for attempt in range(3):
        try:
            if not os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, "w") as f:
                    f.write("# Rezultati modelov — FTIR klasifikacija tkiva\n")
                    f.write(f"# {'-'*90}\n")
                print(f"  -> {RESULTS_FILE} (ustvarjena nova)")
            else:
                print(f"  -> {RESULTS_FILE} (dodana vrstica)")
            with open(RESULTS_FILE, "a") as f:
                f.writelines(lines)
            return
        except OSError as e:
            print(f"  OPOZORILO: pisanje v {RESULTS_FILE} ni uspelo "
                  f"(poskus {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2)
    print(f"  OPOZORILO: {RESULTS_FILE} po 3 poskusih se vedno ni dosegljiv.")


# ---------------------------------------------------------------------------
# Trening — fiksne epohe, EN model (brez ansambla)
# ---------------------------------------------------------------------------
def train_blind(pca_cube, samples, y, device, criterion, final_epochs, batch_size,
                lr, seed, n_channels, num_classes, use_lrn=False,
                optimizer_name="adam", weight_decay=1e-4, sampler=None):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = SpectralDataset(pca_cube, samples, y)
    if sampler is not None:
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    model = SingleStreamCNN(n_channels=n_channels, num_classes=num_classes,
                            use_lrn=use_lrn).to(device)

    if optimizer_name == "adadelta":
        # Clankov recept za prostorsko-spektralni model (2.5.2 #1) -- na tem modelu KOLABIRA
        # (preverjeno: train ll zamrznjen pri ln(5), en sam napovedan
        # razred, tako z weights kot z oversample strategijo). Ostane na
        # voljo za dokumentacijo/ponovitev tega negativnega rezultata.
        optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"  {'Ep':>4}  {'Train ll':>10}  {'LR':>9}")
    t0 = time.time()
    n_bad = 0
    for epoch in range(1, final_epochs + 1):
        model.train()
        total, n_ok = 0.0, 0
        for x, labels in loader:
            x = x.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            if not torch.isfinite(loss):
                n_bad += 1; optimizer.zero_grad(); continue
            loss.backward()
            finite_grads = all(torch.isfinite(p_.grad).all()
                               for p_ in model.parameters() if p_.grad is not None)
            if not finite_grads:
                n_bad += 1; optimizer.zero_grad(); continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
            n_ok += len(labels)
        train_loss = total / max(n_ok, 1)
        print(f"  {epoch:>4}  {train_loss:>10.5f}  {optimizer.param_groups[0]['lr']:>9.2e}")
    print(f"  Treniran v {time.time()-t0:.1f}s"
          + (f"  [OPOZORILO: {n_bad} preskocenih NaN korakov]" if n_bad else ""))
    return model


@torch.no_grad()
def get_logits_full(model, pca_cube, coords, device, batch_size=512):
    model.eval()
    ds = SpectralDataset(pca_cube, coords, np.zeros(len(coords), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    out = []
    for x, _ in loader:
        out.append(model.get_logits(x.to(device)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
def main():
    global NUM_CLASSES, NUM_WORKERS
    run_start = time.time()

    parser = argparse.ArgumentParser(
        description="Spektralni CNN -- arhitektura prostorsko-spektralnega "
                    "modela brez prostorske okolice, clanku zvest recept nadvzorcenja."
    )
    parser.add_argument("--fullslide-dir", default="FTIR-data/fullslide")
    parser.add_argument("--label-set", type=int, choices=[5, 7], default=5)
    parser.add_argument("--output", default="modelB_spectral_cnn_faithful.npz")
    parser.add_argument("--final-epochs", type=int, default=8)
    parser.add_argument("--balance-strategy",
                        choices=["weights", "oversample", "oversample_batch"],
                        default="oversample",
                        help="'oversample' (privzeto) = clankov recept, fiksen pool "
                             "100k/razred. 'weights' = utezen loss (nas izum, ni v "
                             "clanku). 'oversample_batch' = sveze sampling vsako "
                             "epoho (nas izum, testiran, propadel -- 22.36% OA).")
    parser.add_argument("--weight-soften", type=float, default=WEIGHT_SOFTEN,
                        help="Samo za --balance-strategy weights.")
    parser.add_argument("--oversample-target", type=int, default=100000,
                        help="Ciljno stevilo vzorcev na razred pri nadvzorceni (clanek: 100.000).")
    parser.add_argument("--use-lrn", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "adadelta"], default="adam",
                        help="'adam' (privzeto) -- edini, ki na tem modelu deluje. "
                             "'adadelta' (clankov recept za prostorsko-spektralni model, lr=0.1) kolabira.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    NUM_WORKERS = args.num_workers

    print("\n=== 1. Nalaganje celorezinskih PCA podatkov ===")
    d = args.fullslide_dir
    train_pca = np.load(f"{d}/train_pca16.npy")
    test_pca  = np.load(f"{d}/test_pca16.npy")
    train_labels, remap_info = load_labels(f"{d}/train_labels{args.label_set}.npy", args.label_set)
    test_labels, remap_info_test = load_labels(f"{d}/test_labels{args.label_set}.npy", args.label_set)
    assert [r[0] for r in remap_info] == [r[0] for r in remap_info_test], \
        "Train in test labels imata razlicen nabor prisotnih razredov -- prekinjam."
    NUM_CLASSES = len(remap_info)
    n_pca = train_pca.shape[-1]
    print(f"  train_pca16: {train_pca.shape}   test_pca16: {test_pca.shape}")
    print(f"  NUM_CLASSES = {NUM_CLASSES} (label-set={args.label_set})")
    print_remap(remap_info, "train==test")

    device = get_device()
    print(f"\n  Konfiguracija: SPEKTRALNO SAMO (brez prostorske okolice) | n_pca={n_pca} | "
          f"optimizer={args.optimizer} (lr={args.lr}) | balance={args.balance_strategy}")

    print(f"\n=== 2. Vsa anotirana ucna rezina (brez notranje validacije, clanek nima "
          f"opisane validacije za SD) ===")
    coords = np.argwhere(train_labels != -1)
    y_true = train_labels[coords[:, 0], coords[:, 1]].astype(np.int64)
    print(f"  Skupaj train: {len(y_true):,} pikslov")

    weights = compute_class_weights(y_true, NUM_CLASSES, soften=args.weight_soften)
    criterion = build_criterion(args.balance_strategy, weights, device)

    samples, sampler = coords.copy(), None
    if args.balance_strategy == "oversample":
        samples, y = oversample_pool(samples, y_true, NUM_CLASSES, seed=args.seed,
                                     target=args.oversample_target)
    elif args.balance_strategy == "oversample_batch":
        y = y_true
        epoch_len = args.oversample_target or len(y_true)
        sampler = BalancedOnTheFlySampler(y, NUM_CLASSES, epoch_len=epoch_len, seed=args.seed)
    else:
        y = y_true

    print(f"\n=== 3. Trening koncnega modela (seed={args.seed}) ===")
    model = train_blind(
        train_pca, samples, y, device, criterion,
        final_epochs=args.final_epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, n_channels=n_pca, num_classes=NUM_CLASSES,
        use_lrn=args.use_lrn, optimizer_name=args.optimizer,
        weight_decay=args.weight_decay, sampler=sampler,
    )

    print(f"\n=== 4. TEST (locena fizicna rezina, prvi in edini dotik) ===")
    coords_test = np.argwhere(test_labels != -1)
    y_test = test_labels[coords_test[:, 0], coords_test[:, 1]].astype(np.int64)
    print(f"  Skupaj TEST: {len(y_test):,} pikslov")
    test_logits = get_logits_full(model, test_pca, coords_test, device)

    print("\n=== 5. KONCNA evaluacija na TEST ===")
    test_probs = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
    test_probs /= test_probs.sum(axis=1, keepdims=True)
    test_pred = np.argmax(test_probs, axis=1)
    test_oa = accuracy_score(y_test, test_pred)
    test_ll = log_loss(y_test, test_probs, labels=np.arange(NUM_CLASSES))
    print(f"  TEST OA: {test_oa*100:.2f}%")
    print(f"  TEST ll: {test_ll:.5f}")
    print(f"  Ref clanek CNN (spektralni, SD): OA=62.52%")
    print(f"  Ref clanek SVM (SD): OA=56.41%")
    print(f"  Ref clanek CNN (prostorsko-spektralni, SD): OA=79.45%+/-1.25 (79.18% 5-skupnih)")
    print_per_class_table(y_test, test_pred, test_probs, NUM_CLASSES, remap_info,
                          "Rezultati po razredih, OA in log-loss (TEST -- koncni test):")

    np.savez_compressed(args.output, probs=test_probs.astype(np.float32),
                        coords_test=coords_test, y_test=y_test,
                        remap_native_idx=[r[0] for r in remap_info],
                        remap_names=[r[1] for r in remap_info])
    print(f"\n  Shranjeno: {args.output}")

    print("\n=== POVZETEK ===")
    print(f"  SKUPAJ CAS TEKA: {format_duration(time.time() - run_start)}")
    print(f"  Arhitektura: SingleStreamCNN (brez BN, softplus, dropout 0.5, N(0,0.02))")
    print(f"  Optimizer: {args.optimizer} (lr={args.lr})")
    print(f"  Balance: {args.balance_strategy}")
    print(f"  Fiksne epohe: {args.final_epochs} | BREZ ansambla (seed={args.seed})")
    print(f"\n  TEST: OA={test_oa*100:.2f}%  ll={test_ll:.5f}")
    print(f"\n  Primerjava:")
    print(f"    Clanek SVM (SD):                     OA=56.41%")
    print(f"    Clanek CNN (spektralni, SD):            OA=62.52%")
    print(f"    Clanek CNN (prostorsko-spektralni, SD):    OA=79.45%+/-1.25")

    print(f"\n=== 6. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelB_spectral_cnn_faithful",
        test_oa=test_oa, test_ll=test_ll, output_path=args.output,
        final_epochs=args.final_epochs,
        extra_note=(f"spektralni CNN, prostorsko-spektralna arhitektura brez okolice. "
                    f"[balance={args.balance_strategy}, "
                    f"oversample_target={args.oversample_target}, "
                    f"optimizer={args.optimizer}, lr={args.lr}, "
                    f"weight_decay={args.weight_decay}, use_lrn={args.use_lrn}, "
                    f"seed={args.seed}]"),
        total_duration_str=format_duration(time.time() - run_start),
    )


if __name__ == "__main__":
    main()
