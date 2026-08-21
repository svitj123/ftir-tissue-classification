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
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset

PATCH_SIZE   = 17
RESULTS_FILE = "rezultati_report.txt"
NUM_WORKERS  = 0
NUM_CLASSES  = 5          # razreseno dinamicno v main() iz nalozenih oznak

GRID_H, GRID_W = 800, 1200   # ista mreza kot pri starih izsekih, samo za izbiro inner-vala
SIGMA_CHOICES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_SOFTEN = 0.5

# native vrstni red R0..R6 v {train,test}_labels{5,7}.npy (glej build_fullslide_std.py)
NATIVE_NAMES = ["coll", "epith", "fibro", "lymph", "myo", "necrosis", "blood"]

METODOLOGIJA_OPOMBA = (
    "Prostorsko-spektralni CNN, dropout samo na FC plasti, nastavljiv "
    "--dropout in --weight-decay, nadvzorcenje po clanku zvestem receptu. "
    "Ref. clanek: SVM=56.41%, CNN=79.45%+/-1.25 (vseh 6 razredov), 79.18% (5 "
    "skupnih, brez adipocitov)."
)


# ---------------------------------------------------------------------------
# Nalaganje celorezinskih podatkov + remap oznak na strnjene 0..K-1
# ---------------------------------------------------------------------------
def load_labels(path, label_set):
    """Nalozi native-red oznake (R0..R6, -1=neanotirano) in jih remapira na
    strnjene 0..K-1. Vrne (labels_remapped, remap_info) kjer je remap_info
    seznam (native_idx, native_name, new_idx)."""
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
# Izbira inner-val celice (nadomesti select_inner_val_candidates iz crop-ov)
# ---------------------------------------------------------------------------
def select_inner_val_region(labels, num_classes, grid_h=GRID_H, grid_w=GRID_W, verbose=True):
    H, W = labels.shape
    candidates = []
    if verbose:
        print(f"  Pregled {grid_h}x{grid_w} mreznih celic ucne rezine ({H}x{W})...")
    for r0 in range(0, H, grid_h):
        for c0 in range(0, W, grid_w):
            r1, c1 = min(r0 + grid_h, H), min(c0 + grid_w, W)
            cell = labels[r0:r1, c0:c1]
            ann = cell[cell != -1]
            n = int(ann.size)
            if n == 0:
                continue
            vals, counts = np.unique(ann, return_counts=True)
            has_all = len(vals) == num_classes
            min_count = int(counts.min()) if has_all else 0
            candidates.append(dict(r0=r0, c0=c0, r1=r1, c1=c1, n=n,
                                   has_all=has_all, min_count=min_count))

    with_all = [c for c in candidates if c["has_all"]]
    if with_all:
        with_all.sort(key=lambda c: -c["min_count"])
        best = with_all[0]
    else:
        candidates.sort(key=lambda c: -c["n"])
        best = candidates[0]
        if verbose:
            print("  OPOZORILO: nobena celica nima vseh razredov. Vzeta najvecja.")
    if verbose:
        print(f"  Izbran inner-val: vrstice [{best['r0']}:{best['r1']}), "
              f"stolpci [{best['c0']}:{best['c1']}), {best['n']:,} anotiranih, "
              f"min_razred={best['min_count'] if best['has_all'] else '-'}")
    return best["r0"], best["c0"], best["r1"], best["c1"]


def split_inner_train_val(labels, val_bbox, pad):
    """Vrne (coords_it, y_it, coords_iv_local, y_iv, iv_shape).

    coords_it so GLOBALNE koordinate (v celotni ucni rezini) za inner-train,
    IZKLJUCUJE vse anotirane tocke znotraj [r0-pad, r1+pad) x [c0-pad, c1+pad)
    -- da inner-train patchi nikoli ne vidijo inner-val obmocja (enakovredno
    fizicno locenima izsekoma, ki sta bila prej v locenih datotekah).
    coords_iv_local so LOKALNE koordinate znotraj izrezane iv celice."""
    r0, c0, r1, c1 = val_bbox
    H, W = labels.shape
    all_coords = np.argwhere(labels != -1)
    all_y = labels[all_coords[:, 0], all_coords[:, 1]].astype(np.int64)

    rr, cc = all_coords[:, 0], all_coords[:, 1]
    in_buffer = ((rr >= r0 - pad) & (rr < r1 + pad) &
                (cc >= c0 - pad) & (cc < c1 + pad))
    coords_it = all_coords[~in_buffer]
    y_it = all_y[~in_buffer]

    iv_labels_local = labels[r0:r1, c0:c1]
    coords_iv_local = np.argwhere(iv_labels_local != -1)
    y_iv = iv_labels_local[coords_iv_local[:, 0], coords_iv_local[:, 1]].astype(np.int64)

    n_excluded_from_it = int(in_buffer.sum()) - len(y_iv)
    print(f"  inner-train: {len(y_it):,}px (izkljucenih {int(in_buffer.sum()):,} v "
          f"puferju okoli inner-vala, od tega {len(y_iv):,} je sam inner-val)")
    return coords_it, y_it, coords_iv_local, y_iv, (r1 - r0, c1 - c0)


# ---------------------------------------------------------------------------
# Dataset — patch iz ENE celorezinske (ze paddane) PCA slike
# ---------------------------------------------------------------------------
_AUGMENTS = [
    lambda p: p,
    lambda p: torch.flip(p, [1]),
    lambda p: torch.flip(p, [2]),
    lambda p: torch.rot90(p, 1, [1, 2]),
    lambda p: torch.rot90(p, 2, [1, 2]),
    lambda p: torch.rot90(p, 3, [1, 2]),
    lambda p: torch.flip(torch.rot90(p, 1, [1, 2]), [1]),
    lambda p: torch.flip(torch.rot90(p, 1, [1, 2]), [2]),
]


def pad_pca(data_pca, patch_size=PATCH_SIZE):
    pad = patch_size // 2
    return np.pad(data_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect').astype(np.float32)


class PatchDataset(Dataset):
    """samples: (N,2) int array (r,c) GLOBALNIH koordinat znotraj (nepaddane)
    slike, iz katere je bil padded_pca izracunan."""
    def __init__(self, padded_pca, samples, labels, patch_size=PATCH_SIZE,
                augment=False, tta_idx=-1):
        self.pad = patch_size // 2
        self.padded_pca = padded_pca
        self.samples = samples
        self.labels = labels
        self.augment = augment
        self.tta_idx = tta_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r, c = self.samples[idx]
        pad = self.pad
        rp, cp = r + pad, c + pad
        patch = self.padded_pca[rp-pad:rp+pad+1, cp-pad:cp+pad+1]
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1).copy())
        if self.tta_idx >= 0:
            patch_t = _AUGMENTS[self.tta_idx](patch_t)
        elif self.augment:
            patch_t = _AUGMENTS[random.randint(0, 7)](patch_t)
        return patch_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Arhitektura: Single-Stream CNN (clanek Fig. 3, SD varianta = brez BN)
# ---------------------------------------------------------------------------
class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, n_heads=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-4)
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout,
                                          batch_first=True)
        self.proj = nn.Linear(channels, channels)
        self.gamma = nn.Parameter(torch.zeros(channels))

    def zero_init_output(self):
        nn.init.zeros_(self.gamma)

    def forward(self, f):
        B, C, H, W = f.shape
        x = f.flatten(2).transpose(1, 2)
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.gamma * self.proj(a)
        return x.transpose(1, 2).reshape(B, C, H, W)


class SingleStreamCNN(nn.Module):
    def __init__(self, n_channels, num_classes=None, patch_size=PATCH_SIZE,
                dropout=0.5, use_lrn=False, use_attention=False, attn_heads=4):
        super().__init__()
        num_classes = NUM_CLASSES if num_classes is None else num_classes
        lrn = (lambda: nn.LocalResponseNorm(size=5)) if use_lrn else (lambda: nn.Identity())
        # Clanek 2.5.2 #2: "Dropout keeps the activation of a fraction of hidden
        # nodes... randomly turns off the activation of the rest" -- to je opis
        # navadnega po-enoto dropouta (TFLearn privzeto), NE kanalskega/spatial
        # dropouta. nn.Dropout2d (prejsnja razlicica) izklopi CEL KANAL naenkrat,
        # kar je bistveno agresivnejsa, namerna regularizacija, ki je clanek ne
        # omenja -- popravljeno na navaden nn.Dropout povsod.
        # ceil_mode=True -- replika TFLearn/TF privzetega padding='same' za
        # max_pool_2d. TF SAME pooling zaokrozi navzgor (17->9->5),
        # PyTorch privzeto navzdol (17->8->4). ceil_mode=True je matematicno
        # enakovreden TF-jevemu -inf-zapolnjevanju za max operacijo.
        # Dropout je SAMO na FC, odstranjen iz vseh treh konvolucijskih
        # plasti -- clanek ne pove eksplicitno na katerih plasteh je dropout
        # (Fig. 3 ga ne prikaze), obicajna praksa iz casa objave je dropout
        # dajala samo na FC.
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.Softplus(),
            lrn(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Softplus(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Softplus(),
            lrn(),
            nn.MaxPool2d(2, ceil_mode=True),
        )
        after_pool1 = -(-patch_size // 2)   # ceil division brez float
        pooled = -(-after_pool1 // 2)
        # VAROVALKA: prava lazna napoved skozi features, da se morebitna
        # napaka v rocnem izracunu pokaze TAKOJ, ne sele sredi treninga.
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, patch_size, patch_size)
            dummy_out = self.features(dummy)
        assert dummy_out.shape == (1, 64, pooled, pooled), (
            f"pooled izracunan narobe: pricakovano (1,64,{pooled},{pooled}), "
            f"dejansko {tuple(dummy_out.shape)}")
        self.flatten_dim = 64 * pooled * pooled
        self.attn_block = (SpatialSelfAttention(64, n_heads=attn_heads)
                           if use_attention else None)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()
        if self.attn_block is not None:
            self.attn_block.zero_init_output()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_logits(self, patch):
        f = self.features(patch)
        if self.attn_block is not None:
            f = self.attn_block(f)
        f = f.reshape(f.size(0), -1)
        if f.shape[1] != self.flatten_dim:
            raise RuntimeError(
                f"Nepricakovana dimenzija po conv slojih: {f.shape[1]} "
                f"(pricakovano {self.flatten_dim}). Preveri PATCH_SIZE."
            )
        f = F.softplus(self.fc1(f))
        f = self.dropout(f)
        return self.fc2(f)

    def forward(self, patch):
        return self.log_softmax(self.get_logits(patch))


class SmoothedNLLLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.1, n_classes=None):
        super().__init__()
        n_classes = NUM_CLASSES if n_classes is None else n_classes
        self.weight = weight
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, log_probs, target):
        nll = F.nll_loss(log_probs, target, weight=self.weight, reduction='none')
        smooth = -log_probs.mean(dim=1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()


def build_criterion(strategy, weights, device, label_smoothing=0.0):
    unweighted = strategy in ("oversample", "oversample_batch")
    w = None if unweighted else weights.to(device)
    if label_smoothing > 0:
        return SmoothedNLLLoss(weight=w, smoothing=label_smoothing)
    if unweighted:
        return nn.NLLLoss()
    else:
        return nn.NLLLoss(weight=w)


# ---------------------------------------------------------------------------
# Balansiranje razredov
# ---------------------------------------------------------------------------
def oversample_pool(samples, y, num_classes, seed=42, verbose=True, label="train", target=None):
    rng = np.random.RandomState(seed)
    counts = np.bincount(y, minlength=num_classes)
    target = int(counts.max()) if target is None else int(target)
    if verbose:
        print(f"  Nadvzorcenje ({label}): {list(counts)} -> {target}/razred")
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
    if verbose:
        print(f"  Skupaj po nadvzorcenju ({label}): {len(idx):,} (prej {len(y):,})")
    return samples[idx], y[idx]


class BalancedOnTheFlySampler(torch.utils.data.Sampler):
    """Per-korak razredno uravnotezeno vzorcenje BREZ fiksnega pool-a.

    Za vsak vzorec v epohi: najprej nakljucno izberi razred (uniformno med
    prisotnimi razredi), nato nakljucno izberi ORIGINALNI indeks tega
    razreda (z vracanjem, iz `class_indices[c]`). Noben fiksni seznam
    kopij se NE gradi -- vsaka epoha (vsak klic __iter__) izzreba SVEZE
    zaporedje, zato se manjsinski piksli ne "zamrznejo" na isto stevilo
    ponovitev kot pri oversample_pool().

    epoch_len doloca stevilo vzorcev na epoho (privzeto = stevilo
    originalnih anotiranih pikslov, da epoha stane priblizno enako kot
    pri --balance-strategy weights).
    """
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


def compute_class_weights(y, num_classes, soften=WEIGHT_SOFTEN):
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    raw     = len(y) / (num_classes * np.where(counts > 0, counts, 1))
    weights = raw ** soften
    print(f"  Utezi razredov (soften={soften}): {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Helpers — NESPREMENJENO
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda"); print(f"  Naprava: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        d = torch.device("mps"); print("  Naprava: MPS")
    else:
        d = torch.device("cpu");  print("  Naprava: CPU")
    return d


@torch.no_grad()
def get_logits_array(model, dataset, device, batch_size=512):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    all_l  = []
    for patches, _ in loader:
        all_l.append(model.get_logits(patches.to(device)).cpu().numpy())
    return np.concatenate(all_l)


@torch.no_grad()
def get_logits_tta(model, padded_pca, samples, labels, device, batch_size=512,
                   patch_size=PATCH_SIZE):
    logits_sum = None
    for aug_idx in range(8):
        ds = PatchDataset(padded_pca, samples, labels, patch_size=patch_size,
                          augment=False, tta_idx=aug_idx)
        logits = get_logits_array(model, ds, device, batch_size)
        logits_sum = logits.copy() if logits_sum is None else logits_sum + logits
    return logits_sum / 8


def get_logits_maybe_tta(model, padded_pca, samples, labels, device, use_tta,
                         batch_size=512, patch_size=PATCH_SIZE):
    """use_tta=False: ena sama neobrnjena napoved (clanek TTA ne omenja)."""
    if use_tta:
        return get_logits_tta(model, padded_pca, samples, labels, device,
                              batch_size=batch_size, patch_size=patch_size)
    ds = PatchDataset(padded_pca, samples, labels, patch_size=patch_size,
                      augment=False, tta_idx=-1)
    return get_logits_array(model, ds, device, batch_size)


def find_temperature(logits, y, t_floor=1.0):
    def neg_ll(T):
        s = logits / T
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-9, 1.0)
        return -np.mean(np.log(p[np.arange(len(y)), y]))
    res = minimize_scalar(neg_ll, bounds=(t_floor, 50.0), method='bounded')
    T   = res.x
    print(f"  Temperature (skalarna): T={T:.4f}  {neg_ll(1.0):.5f} -> {neg_ll(T):.5f}")
    return T


def apply_temperature(logits, T):
    s = logits / T
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def find_temperature_per_class(logits, y, n_classes, t_floor=1.0):
    def neg_ll(T_vec):
        T_vec = np.clip(T_vec, t_floor, 60.0)
        s = logits / T_vec[None, :]
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = np.clip(e / e.sum(axis=1, keepdims=True), 1e-9, 1.0)
        return -np.mean(np.log(p[np.arange(len(y)), y]))
    x0 = np.ones(n_classes)
    bounds = [(t_floor, 50.0)] * n_classes
    res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
    T_vec = res.x
    print(f"  Temperature (per-class): " +
          ", ".join(f"R{c}={t:.2f}" for c, t in enumerate(T_vec)))
    print(f"  ll: {neg_ll(np.ones(n_classes)):.5f} -> {neg_ll(T_vec):.5f}")
    return T_vec


def apply_temperature_per_class(logits, T_vec):
    s = logits / T_vec[None, :]
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def format_T(T_opt):
    if np.isscalar(T_opt):
        return f"{float(T_opt):.4f}"
    return "[" + ",".join(f"{t:.3f}" for t in T_opt) + "]"


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h}h {m}min {s}s"
    if m: return f"{m}min {s}s"
    return f"{s}s"


def gaussian_smooth_probs(probs, tissue_mask, sigma):
    if sigma <= 0: return probs
    smoothed = np.zeros_like(probs)
    mf = tissue_mask.astype(np.float32)
    for c in range(probs.shape[-1]):
        num = gaussian_filter(probs[:, :, c] * mf, sigma=sigma)
        den = gaussian_filter(mf, sigma=sigma)
        smoothed[:, :, c] = num / np.where(den < 1e-8, 1e-8, den)
    smoothed = np.clip(smoothed, 1e-7, 1.0)
    smoothed /= smoothed.sum(axis=-1, keepdims=True)
    return smoothed.astype(np.float32)


def build_full_canvas_prob_map(H, W, coords, probs, fallback_prior, num_classes):
    prob_map = np.tile(fallback_prior, (H * W, 1)).reshape(H, W, num_classes)
    prob_map[coords[:, 0], coords[:, 1]] = probs
    return prob_map


def find_best_sigma(H, W, coords, probs, y_true, tissue_mask, fallback_prior,
                    num_classes, sigma_choices=SIGMA_CHOICES, select_by="logloss"):
    prob_map = build_full_canvas_prob_map(H, W, coords, probs, fallback_prior, num_classes)
    print(f"  {'sigma':>6}  {'log-loss':>10}  {'CA':>8}   (izbira po: {select_by})")
    best_sigma, best_score = 0.0, None
    for sigma in sigma_choices:
        if sigma > 0:
            smoothed = gaussian_smooth_probs(prob_map, tissue_mask, sigma)
            m = prob_map.copy()
            m[tissue_mask] = smoothed[tissue_mask]
        else:
            m = prob_map
        probs_at_coords = m[coords[:, 0], coords[:, 1]]
        probs_at_coords = np.clip(probs_at_coords, 1e-7, 1.0)
        probs_at_coords /= probs_at_coords.sum(axis=1, keepdims=True)
        ll = log_loss(y_true, probs_at_coords, labels=np.arange(num_classes))
        ca = accuracy_score(y_true, np.argmax(probs_at_coords, axis=1))
        score = ca if select_by == "ca" else -ll
        marker = ""
        if best_score is None or score > best_score:
            best_score = score; best_sigma = sigma; marker = " *"
        print(f"  {sigma:>6.1f}  {ll:>10.5f}  {ca*100:>7.2f}%{marker}")
    print(f"  Najboljsa sigma: {best_sigma:.1f}  (izbrana po {select_by})")
    return best_sigma


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


def write_results_report(model_name, innerval_oa, innerval_ll, test_oa, test_ll,
                          output_path, t_opt_str, sigma, final_epochs,
                          extra_note="", total_duration_str=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dur_part = f"  cas={total_duration_str}" if total_duration_str else ""
    lines = [
        f"{timestamp}  {model_name:<25}  "
        f"INNERVAL_OA={innerval_oa*100:6.2f}%  INNERVAL_ll={innerval_ll:.5f}  "
        f"TEST_OA={test_oa*100:6.2f}%  TEST_ll={test_ll:.5f}  "
        f"T={t_opt_str}  sigma={sigma:.1f}  n_ensemble=1  "
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
    print(f"  OPOZORILO: {RESULTS_FILE} po 3 poskusih se vedno ni dosegljiv. "
          f"Vsebina (dodaj rocno, ce zelis):")
    print("  " + "".join(lines).replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Trening — fiksne epohe, EN model (brez ansambla)
# ---------------------------------------------------------------------------
def train_blind(padded_pca, samples, y, device, criterion,
                final_epochs, batch_size, lr, seed, n_channels,
                num_classes, patch_size=PATCH_SIZE, use_lrn=False,
                use_attention=False, attn_heads=4, optimizer_name="adam",
                weight_decay=1e-4, sampler=None, augment=True, grad_clip=1.0,
                epoch_callback=None, dropout=0.5):
    torch.manual_seed(seed); random.seed(seed)
    train_ds = PatchDataset(padded_pca, samples, y, patch_size=patch_size, augment=augment)
    if sampler is not None:
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    model     = SingleStreamCNN(n_channels=n_channels, num_classes=num_classes,
                                patch_size=patch_size, use_lrn=use_lrn, dropout=dropout,
                                use_attention=use_attention, attn_heads=attn_heads).to(device)
    # Clanek 2.5.2 #1: "Adadelta adaptive learning rate method with lr=0.1" --
    # Adadelta sam prilagaja efektivno hitrost ucenja, zato clanek NE omenja
    # nobenega razporejevalnika. Cosine decay je bil dodan SAMO ob Adamu
    # (nasa modernizacija) -- pri Adadelta ga zato izpustimo, da ostane teden
    # zvest clanku.
    # weight_decay: Berisha in sod. (2019) jakosti L2 ne pove. Lotfollahi in
    # sod. (2019), ista raziskovalna skupina, navaja L2=0.001 -- verjeten
    # kandidat za dejansko (nerazkrito) vrednost v nasem clanku, ker skupine
    # pogosto ponovno uporabijo iste privzete vrednosti med sorodnimi
    # projekti.
    if optimizer_name == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
        if abs(lr - 1e-3) < 1e-9:
            print(f"  OPOZORILO: --optimizer adadelta z lr=1e-3 (Adam privzeta vrednost) je "
                  f"verjetno prenizka za Adadelta -- clanek uporablja lr=0.1.")
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs, eta_min=1e-6)

    print(f"  {'Ep':>4}  {'Train ll':>10}  {'LR':>9}")
    t0 = time.time()
    n_bad = 0
    for epoch in range(1, final_epochs + 1):
        model.train()
        total = 0.0
        n_ok = 0
        for patches, labels in loader:
            patches = patches.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(patches), labels)
            if not torch.isfinite(loss):
                n_bad += 1
                optimizer.zero_grad()
                continue
            loss.backward()
            finite_grads = all(torch.isfinite(p_.grad).all()
                               for p_ in model.parameters() if p_.grad is not None)
            if not finite_grads:
                n_bad += 1
                optimizer.zero_grad()
                continue
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item() * len(labels)
            n_ok += len(labels)
        train_loss = total / max(n_ok, 1)
        if scheduler is not None:
            scheduler.step()
        diag = ""
        if epoch_callback is not None:
            diag = epoch_callback(model, epoch)
        print(f"  {epoch:>4}  {train_loss:>10.5f}  "
              f"{optimizer.param_groups[0]['lr']:>9.2e}{diag}")
    print(f"  Treniran v {time.time()-t0:.1f}s"
          + (f"  [OPOZORILO: {n_bad} preskocenih NaN korakov]" if n_bad else ""))
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global NUM_WORKERS, NUM_CLASSES
    run_start = time.time()

    parser = argparse.ArgumentParser(
        description="Model C Full-Slide: single-stream CNN po clanku, brez razreza "
                    "na izseke, brez ansambla (en seed + TTA)."
    )
    parser.add_argument("--fullslide-dir", default="FTIR-data/fullslide")
    parser.add_argument("--label-set", type=int, choices=[5, 7], default=5,
                        help="5 = clanku-primerljivi razredi (fibro/lymph deanotirana, "
                             "privzeto). 7 = vsi razredi.")
    parser.add_argument("--output", default="modelC_fullslide.npz")
    parser.add_argument("--final-epochs", type=int, default=8,
                        help="Fiksno stevilo epoh (clanek: 8, tocka 2.5.2 #8).")
    parser.add_argument("--per-class-temperature", action="store_true")
    parser.add_argument("--no-temperature", action="store_true",
                        help="Izklopi temperaturno umerjanje POPOLNOMA (T=1.0 fiksno, "
                             "surov softmax) -- clanek tega ne omenja. Preglasi "
                             "--per-class-temperature.")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Izklopi prostorsko glajenje POPOLNOMA (sigma=0.0 fiksno, "
                             "brez sweep-a) -- clanek tega ne omenja.")
    parser.add_argument("--no-tta", action="store_true",
                        help="Izklopi 8x TTA, uporabi eno samo (neobrnjeno) napoved "
                             "na model -- clanek TTA ne omenja.")
    parser.add_argument("--balance-strategy",
                        choices=["weights", "oversample", "oversample_batch"],
                        default="oversample",
                        help="'oversample' (privzeto za to datoteko) = clankov "
                             "dobesedni recept (100k/razred, identicno podvajanje). "
                             "Memorizacija se blazi z --dropout/--weight-decay, ne "
                             "z vecjo raznolikostjo vzorcev. 'weights' = nas izum.")
    parser.add_argument("--weight-soften", type=float, default=WEIGHT_SOFTEN)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--oversample-target", type=int, default=100000)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout samo na FC plasti, brez konvolucijskih. Clanek "
                             "navaja keep=0.5 (dropout=0.5), a ne pove ali je to "
                             "primerno tudi za oversampled (500k, mocno ponovljen) "
                             "train set -- vecji dropout (0.6-0.7) je standarden "
                             "recept proti memorizaciji.")
    parser.add_argument("--skip-faza-a", action="store_true",
                        help="Izpusti Faza A (inner-val kalibracijo) POPOLNOMA -- "
                             "clanek za SD ne opisuje nobene validacijske mnozice. "
                             "Vsili T=1.0 in sigma=0.0 (isto kot --no-temperature "
                             "--no-smoothing), a brez treninga kalibracijskega modela "
                             "-- prihrani ~cas ene cele epohe-runde.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Izklopi D4 (8x rotacije/zrcaljenja) augmentacijo patchev "
                             "med treningom. Clanek (2.5.2 #9) omenja SAMO nakljucno "
                             "mesanje vrstnega reda vzorcev na epoho, NE augmentacije "
                             "vsebine -- to je bilo prej nedokumentirano odstopanje.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping (max L2 norm). Clanek ga NE omenja. "
                             "Tako TFLearn-ov privzetek 5.0 kot popolen izklop (0) sta "
                             "dala slabsi rezultat od nase izvirne 1.0 (79.63% pri 1.0 "
                             "proti 70.70% pri 5.0 in 64.28% pri 0) -- 1.0 zato ostaja "
                             "privzeta vrednost.")
    parser.add_argument("--extra-smooth-scale", type=str, default="",
                        help="Vejica-loceni seznam skal za dodatne glajene PCA "
                             "kanale (npr. '3').")
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="clanek (Adadelta): lr=0.1. Privzeto 1e-3 je za Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularizacija. Berisha in sod. (2019) jakosti ne "
                             "povejo. Lotfollahi in sod. (2019, ista raziskovalna "
                             "skupina) navajajo L2=0.001 -- verjeten kandidat za "
                             "dejansko vrednost.")
    parser.add_argument("--optimizer", choices=["adam", "adadelta"], default="adam",
                        help="clanek 2.5.2 #1: Adadelta, lr=0.1, BREZ razporejevalnika "
                             "(sam prilagaja hitrost ucenja). 'adam' (privzeto) = nasa "
                             "modernizacija, Adam+cosine decay.")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--use-lrn", action="store_true")
    parser.add_argument("--select-by", choices=["logloss", "ca"], default="logloss")
    parser.add_argument("--t-floor", type=float, default=1.0)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--grid-h", type=int, default=GRID_H)
    parser.add_argument("--grid-w", type=int, default=GRID_W)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extra_smooth_scale = ([int(s) for s in args.extra_smooth_scale.split(",") if s.strip()]
                          if args.extra_smooth_scale.strip() else [])
    NUM_WORKERS = args.num_workers
    patch_size = args.patch_size
    pad = patch_size // 2

    # ------------------------------------------------------------------
    print("\n=== 1. Nalaganje celorezinskih podatkov ===")
    d = args.fullslide_dir
    train_pca = np.load(f"{d}/train_pca16.npy")
    test_pca  = np.load(f"{d}/test_pca16.npy")
    train_tissue = np.load(f"{d}/train_tissue.npy")
    test_tissue  = np.load(f"{d}/test_tissue.npy")
    train_labels, remap_info = load_labels(f"{d}/train_labels{args.label_set}.npy", args.label_set)
    test_labels, remap_info_test = load_labels(f"{d}/test_labels{args.label_set}.npy", args.label_set)
    assert train_tissue.shape == train_pca.shape[:2], \
        f"train_tissue {train_tissue.shape} != train_pca {train_pca.shape[:2]}"
    assert test_tissue.shape == test_pca.shape[:2], \
        f"test_tissue {test_tissue.shape} != test_pca {test_pca.shape[:2]}"
    assert [r[0] for r in remap_info] == [r[0] for r in remap_info_test], \
        "Train in test labels imata razlicen nabor prisotnih razredov -- prekinjam."
    NUM_CLASSES = len(remap_info)
    print(f"  train_pca16: {train_pca.shape}   test_pca16: {test_pca.shape}")
    print(f"  NUM_CLASSES = {NUM_CLASSES} (label-set={args.label_set})")
    print_remap(remap_info, "train==test")

    n_channels = train_pca.shape[-1] * (1 + len(extra_smooth_scale))
    if extra_smooth_scale:
        print(f"\n  Extra-smooth-scale {extra_smooth_scale}: dodajam glajene PCA kanale...")
        def add_smooth(cube):
            parts = [cube]
            for scale in extra_smooth_scale:
                parts.append(uniform_filter(cube, size=[scale, scale, 1], mode='reflect'))
            return np.concatenate(parts, axis=-1).astype(np.float32)
        train_pca = add_smooth(train_pca)
        test_pca = add_smooth(test_pca)
        print(f"  train_pca16 -> {train_pca.shape}   test_pca16 -> {test_pca.shape}")

    device = get_device()
    n_param = sum(p.numel() for p in SingleStreamCNN(
        n_channels=n_channels, num_classes=NUM_CLASSES, patch_size=patch_size,
        use_lrn=args.use_lrn, use_attention=args.use_attention,
        attn_heads=args.attn_heads).parameters() if p.requires_grad)
    print(f"\n  Konfiguracija: Patch {patch_size}x{patch_size} | kanalov={n_channels} | "
          f"TTA {'izklopljen' if args.no_tta else '8x'} | "
          f"D4 augment {'IZKLOPLJEN (--no-augment)' if args.no_augment else 'vklopljen'} | "
          f"grad-clip={'IZKLOPLJEN' if args.grad_clip <= 0 else args.grad_clip} | "
          f"dropout={args.dropout} | weight_decay={args.weight_decay} | "
          f"BREZ ansambla (seed={args.seed}) | "
          f"optimizer={args.optimizer} "
          f"(lr={args.lr}) | SingleStreamCNN ({n_param:,} param)")
    print(f"\n  Metodologija: {METODOLOGIJA_OPOMBA}")

    final_epochs = args.final_epochs

    # ==================================================================
    # FAZA A — kalibracija na inner-val celici znotraj ucne rezine
    # ==================================================================
    if args.skip_faza_a:
        print("\n=== 2-6. Faza A IZPUSCENA (--skip-faza-a) ===")
        print("  Clanek za SD ne opisuje nobene lokalne validacijske mnozice --")
        print("  gremo direktno na Faza B. T=1.0 fiksno, sigma=0.0 fiksno.")
        T_opt = 1.0
        apply_T = lambda logits: apply_temperature(logits, 1.0)
        T_opt_str = format_T(T_opt)
        sigma_opt = 0.0
        innerval_oa = float("nan")
        innerval_ll = float("nan")
    else:
        print(f"\n=== 2. Izbira inner-val celice ({args.grid_h}x{args.grid_w} mreza) ===")
        val_bbox = select_inner_val_region(train_labels, NUM_CLASSES, args.grid_h, args.grid_w)
        r0, c0, r1, c1 = val_bbox

        print(f"\n=== 3. Faza A — inner-train / inner-val split (pufer={pad}px) ===")
        coords_it, y_it_true, coords_iv, y_iv, iv_shape = split_inner_train_val(
            train_labels, val_bbox, pad)
        print(f"\n  Porazdelitev razredov:")
        for c in range(NUM_CLASSES):
            nt = int((y_it_true == c).sum()); nv = int((y_iv == c).sum())
            print(f"    R{c}: it={nt:6d} ({100*nt/len(y_it_true):.1f}%)  "
                  f"iv={nv:6d} ({100*nv/max(len(y_iv),1):.1f}%)")

        weights_A = compute_class_weights(y_it_true, NUM_CLASSES, soften=args.weight_soften)
        criterion_A = build_criterion(args.balance_strategy, weights_A, device,
                                      label_smoothing=args.label_smoothing)

        padded_train = pad_pca(train_pca, patch_size)
        samples_it = coords_it.copy()
        if args.balance_strategy == "oversample":
            samples_it, y_it = oversample_pool(
                samples_it, y_it_true, NUM_CLASSES, seed=args.seed,
                target=args.oversample_target, label="Faza A")
        else:
            y_it = y_it_true

        # inner-val koordinate so LOKALNE znotraj celice -- za diagnostiko med
        # treningom rabimo GLOBALNE koordinate v padded_train.
        coords_iv_global_diag = coords_iv + np.array([r0, c0])

        def _diag_callback(model, epoch):
            """Ce inner-val ll narasca medtem ko train ll pada, je to znak
            prostorske memorizacije oversampled train seta."""
            logits = get_logits_array(
                model, PatchDataset(padded_train, coords_iv_global_diag, y_iv,
                                    patch_size=patch_size), device)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            oa = accuracy_score(y_iv, np.argmax(probs, axis=1))
            ll = log_loss(y_iv, probs, labels=np.arange(NUM_CLASSES))
            return f"   |  inner-val: OA={oa*100:5.2f}%  ll={ll:.4f}"

        print(f"\n=== 4. Faza A — SAMO DIAGNOSTIKA memorizacije (seed={args.seed}) ===")
        print("  (inner-val se izracuna VSAKO epoho -- ce ll tu narasca medtem ko")
        print("   train ll pada, je to znak memorizacije. NI uporabljeno za T/sigma")
        print("   kalibracijo -- ta ostaneta fiksna T=1.0/sigma=0.0.)")
        model_a = train_blind(
            padded_train, samples_it, y_it, device, criterion_A,
            final_epochs=final_epochs, batch_size=args.batch_size, lr=args.lr,
            seed=args.seed, n_channels=n_channels, num_classes=NUM_CLASSES,
            patch_size=patch_size, use_lrn=args.use_lrn,
            use_attention=args.use_attention, attn_heads=args.attn_heads,
            optimizer_name=args.optimizer, weight_decay=args.weight_decay,
            augment=not args.no_augment, grad_clip=args.grad_clip,
            epoch_callback=_diag_callback, dropout=args.dropout,
        )
        del padded_train, model_a

        # Ta datoteka NE kalibrira T/sigma na podlagi Faze A (namenoma) --
        # Faza A je tu SAMO diagnostika memorizacije, ne izbira hiperparametrov.
        T_opt = 1.0
        apply_T = lambda logits: apply_temperature(logits, 1.0)
        T_opt_str = format_T(T_opt)
        sigma_opt = 0.0
        innerval_oa = float("nan")
        innerval_ll = float("nan")

        del coords_it, y_it_true, coords_iv, y_iv

    # ==================================================================
    # FAZA B — celotna ucna rezina, en model, en dotik s testom
    # ==================================================================
    print(f"\n=== 7. Faza B — vsa anotirana ucna rezina ===")
    coords_ot = np.argwhere(train_labels != -1)
    y_ot_true = train_labels[coords_ot[:, 0], coords_ot[:, 1]].astype(np.int64)
    print(f"  Skupaj Faza B train: {len(y_ot_true):,} pikslov")

    weights_B = compute_class_weights(y_ot_true, NUM_CLASSES, soften=args.weight_soften)
    criterion_B = build_criterion(args.balance_strategy, weights_B, device,
                                  label_smoothing=args.label_smoothing)
    samples_ot = coords_ot.copy()
    sampler_ot = None
    if args.balance_strategy == "oversample":
        samples_ot, y_ot = oversample_pool(
            samples_ot, y_ot_true, NUM_CLASSES, seed=args.seed,
            target=args.oversample_target, label="Faza B")
    elif args.balance_strategy == "oversample_batch":
        y_ot = y_ot_true
        epoch_len = args.oversample_target if args.oversample_target else len(y_ot_true)
        sampler_ot = BalancedOnTheFlySampler(y_ot, NUM_CLASSES, epoch_len=epoch_len,
                                             seed=args.seed)
        print(f"  BalancedOnTheFlySampler (Faza B): {list(np.bincount(y_ot_true, minlength=NUM_CLASSES))} "
              f"-> ~{epoch_len // NUM_CLASSES:,}/razred/epoho (SVEZE vsako epoho, brez fiksnega poola)")
    else:
        y_ot = y_ot_true

    padded_train_full = pad_pca(train_pca, patch_size)
    print(f"\n=== 8. Faza B — trening koncnega modela (seed={args.seed}) ===")
    model_b = train_blind(
        padded_train_full, samples_ot, y_ot, device, criterion_B,
        final_epochs=final_epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, n_channels=n_channels, num_classes=NUM_CLASSES,
        patch_size=patch_size, use_lrn=args.use_lrn,
        use_attention=args.use_attention, attn_heads=args.attn_heads,
        optimizer_name=args.optimizer, weight_decay=args.weight_decay,
        sampler=sampler_ot, augment=not args.no_augment, grad_clip=args.grad_clip,
        dropout=args.dropout,
    )
    del padded_train_full, train_pca

    print(f"\n=== 9. Faza B — TEST (locena fizicna rezina, prvi in edini dotik) ===")
    coords_test = np.argwhere(test_labels != -1)
    y_test = test_labels[coords_test[:, 0], coords_test[:, 1]].astype(np.int64)
    print(f"  Skupaj TEST: {len(y_test):,} pikslov")
    padded_test = pad_pca(test_pca, patch_size)
    test_logits = get_logits_maybe_tta(model_b, padded_test, coords_test, y_test, device,
                                       use_tta=not args.no_tta, patch_size=patch_size)
    del model_b

    # ------------------------------------------------------------------
    print("\n=== 10. KONCNA evaluacija na TEST ===")
    test_probs = apply_T(test_logits.astype(np.float32))
    test_pred = np.argmax(test_probs, axis=1)
    test_oa   = accuracy_score(y_test, test_pred)
    test_ll   = log_loss(y_test, test_probs, labels=np.arange(NUM_CLASSES))
    print(f"  TEST OA (pred smoothing): {test_oa*100:.2f}%")
    print(f"  TEST ll (pred smoothing): {test_ll:.5f}")
    print(f"  Ref clanek CNN (SD, 6 razr.): OA=79.45% +/- 1.25")
    print(f"  Ref clanek CNN (SD, 5 skupnih, brez adipocitov): OA=79.18%")
    print(f"  Ref clanek SVM (SD, isti split): OA=56.41%")
    print_per_class_table(y_test, test_pred, test_probs, NUM_CLASSES, remap_info,
                          "Rezultati po razredih, OA in log-loss (TEST -- koncni test):")

    print(f"\n=== 11. Prostorsko glajenje TEST (ENKRAT cez celo rezino, sigma={sigma_opt:.1f}) ===")
    prior_B = np.bincount(y_ot_true, minlength=NUM_CLASSES).astype(np.float32)
    prior_B /= prior_B.sum()
    H_test, W_test = test_labels.shape
    prob_map = build_full_canvas_prob_map(H_test, W_test, coords_test, test_probs,
                                          prior_B, NUM_CLASSES)
    if sigma_opt > 0:
        smoothed = gaussian_smooth_probs(prob_map, test_tissue, sigma_opt)
        final_map = prob_map.copy()
        final_map[test_tissue] = smoothed[test_tissue]
    else:
        final_map = prob_map
    final_map = np.clip(final_map, 1e-7, 1.0)
    final_map /= final_map.sum(axis=-1, keepdims=True)

    at_coords = final_map[coords_test[:, 0], coords_test[:, 1]]
    at_coords = np.clip(at_coords, 1e-7, 1.0)
    at_coords /= at_coords.sum(axis=1, keepdims=True)
    test_oa_sm = accuracy_score(y_test, np.argmax(at_coords, axis=1))
    test_ll_sm = log_loss(y_test, at_coords, labels=np.arange(NUM_CLASSES))
    print(f"  TEST OA (po smoothing): {test_oa_sm*100:.2f}%")
    print(f"  TEST ll (po smoothing): {test_ll_sm:.5f}")

    np.savez_compressed(args.output, probs=final_map.astype(np.float32),
                        coords_test=coords_test, y_test=y_test,
                        remap_native_idx=[r[0] for r in remap_info],
                        remap_names=[r[1] for r in remap_info])
    print(f"\n  Shranjeno: {args.output}")

    # ------------------------------------------------------------------
    print("\n=== POVZETEK (fullslide) ===")
    print(f"  SKUPAJ CAS TEKA: {format_duration(time.time() - run_start)}")
    print(f"  Train: {len(y_ot_true):,} pikslov (br1003-br2085b, cela rezina)")
    print(f"  Test:  {len(y_test):,} pikslov (brc961-br1001, cela rezina)")
    print(f"  Arhitektura: SingleStreamCNN (brez BN, softplus, dropout 0.5, N(0,0.02))")
    print(f"  Balance strategy: {args.balance_strategy} | Temperature: "
          f"{'per-class' if args.per_class_temperature else 'skalarna'} ({T_opt_str})")
    print(f"  Fiksne epohe: {final_epochs} | BREZ ansambla (seed={args.seed})")
    print(f"  sigma={sigma_opt:.1f}")
    print(f"\n  Inner-val:              OA={innerval_oa*100:.2f}%  ll={innerval_ll:.5f}")
    print(f"  Faza B (TEST, KONCNI): OA={test_oa_sm*100:.2f}%  ll={test_ll_sm:.5f}")

    print(f"\n=== 12. Zapis v {RESULTS_FILE} ===")
    write_results_report(
        model_name="modelC_fullslide",
        innerval_oa=innerval_oa, innerval_ll=innerval_ll,
        test_oa=test_oa_sm, test_ll=test_ll_sm,
        output_path=args.output, t_opt_str=T_opt_str, sigma=sigma_opt,
        final_epochs=final_epochs,
        total_duration_str=format_duration(time.time() - run_start),
        extra_note=(METODOLOGIJA_OPOMBA +
                   f" [strategy={args.balance_strategy}, "
                   f"oversample_target={args.oversample_target}, "
                   f"per_class_T={args.per_class_temperature}, "
                   f"patch_size={patch_size}, use_lrn={args.use_lrn}, "
                   f"use_attention={args.use_attention}, "
                   f"extra_smooth_scale={extra_smooth_scale}, "
                   f"weight_soften={args.weight_soften}, seed={args.seed}, "
                   f"label_set={args.label_set}, optimizer={args.optimizer}, "
                   f"lr={args.lr}, no_temperature={args.no_temperature}, "
                   f"no_smoothing={args.no_smoothing}, no_tta={args.no_tta}]")
    )


if __name__ == "__main__":
    main()
