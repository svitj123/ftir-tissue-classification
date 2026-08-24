# FTIR Tissue Classification

Koda za diplomsko nalogo *Globoke nevronske mreže za klasifikacijo
spektralno-prostorskih podatkov FTIR* (Fakulteta za računalništvo in
informatiko, Univerza v Ljubljani). Naloga čim zvesteje ponovi tri
klasifikacijske pristope iz Berisha in sod. (2019), *"Deep learning for
FTIR histology: leveraging spatial and spectral features with
convolutional neural networks"* (The Analyst), na pravem medrezinskem
razdelku podatkov (dve fizično ločeni tkivni rezini), in sistematično
razreši vrsto ključnih hiperparametrov, ki jih članek ne navaja
eksplicitno.

## Rezultati

Skupna klasifikacijska točnost (OA) na testni rezini, 10 neodvisnih
ponovitev (povprečje ± standardni odklon):

| Model | Naš OA | OA članka |
|---|---|---|
| RBF SVM | 56,55 % ± 0,28 | 56,41 % ± 0,27 |
| Spektralni CNN | 69,98 % ± 3,34 | 62,52 % |
| Prostorsko-spektralni CNN (flagship) | **81,56 % ± 3,18** | 79,18 %* |

\* Članek za spatio-spektralni CNN poroča 79,45 % ± 1,25 na vseh 6
razredih; 79,18 % je preračunano na 5 razredov, skupnih z našimi podatki
(brez adipocitov, glej spodaj).

## Struktura repozitorija

```
ftir-tissue-classification/
├── README.md
├── LICENSE                    (GPL-3.0)
├── requirements.txt
├── models/
│   ├── modelA_svm_rbf_faithful.py        -- RBF SVM
│   ├── modelB_spectral_cnn_faithful.py   -- spektralni CNN
│   └── modelC_fullslide_faithful_v20_v63_regularized.py  -- prostorsko-spektralni CNN (flagship)
├── preprocessing/
│   └── build_fullslide_std.py            -- predobdelava (PCA, predobdelava spektrov)
└── results/
    └── figures/
        ├── *.py                          -- skripte za grafe/tabele iz diplome
        ├── modelA_svm_rbf_faithful_flagship.npz      -- napovedi flagship SVM (testna množica)
        ├── modelA_svm_rbf_faithful_flagship.joblib   -- natreniran flagship SVM model
        ├── modelSpectral_reusedCNN_oversample_seed42.npz  -- napovedi flagship spektralnega CNN
        └── modelC_fullslide_faithful_v20_gc05_seed42.npz  -- napovedi flagship prostorsko-spektralnega CNN
```

## Namestitev

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Testirano s Python 3.11+.

## Podatki

**Podatki niso del tega repozitorija in niso javno dostopni.** Uporabljene
so tkivne mikromreže (TMA) iz Berisha in sod. (2019) -- oznake BR1003,
BR2085b (učna rezina) ter BR961, BR1001 (testna rezina). Mentor jih je od
avtorjev članka pridobil na lastno prošnjo za namen te naloge; za dostop
do surovih podatkov se obrnite neposredno na avtorje izvirnega članka ali
na mentorja te naloge.

Skripte pričakujejo predobdelane podatke (glej `preprocessing/build_fullslide_std.py`)
v naslednji strukturi:

```
FTIR-data/fullslide/
├── train_pca16.npy       -- 16-kan. PCA projekcije, učna rezina
├── test_pca16.npy        -- 16-kan. PCA projekcije, testna rezina
├── train_labels5.npy     -- oznake, 5 skupnih razredov (kolagen, epitelij, miofibroblasti, nekroza, kri)
├── test_labels5.npy
├── train_tissue.npy      -- maska tkiva (bool)
└── test_tissue.npy
```

Naši surovi podatki nimajo razreda adipocitov (v izvornih anotacijah ga
ni), zato je primerjava s člankovimi 6-razrednimi številkami okvirna, ne
stroga -- podrobneje v diplomskem besedilu.

## Zagon modelov

```bash
# RBF SVM (10 ponovitev, ~5h za flagship potrditev)
python3 models/modelA_svm_rbf_faithful.py \
    --save-probs results/figures/modelA_svm_rbf_faithful_flagship.npz \
    --save-model results/figures/modelA_svm_rbf_faithful_flagship.joblib

# Spektralni CNN
python3 models/modelB_spectral_cnn_faithful.py --seed 42 \
    --output results/figures/modelB_spectral_cnn_faithful_seed42.npz

# Prostorsko-spektralni CNN (flagship konfiguracija)
python3 models/modelC_fullslide_faithful_v20_v63_regularized.py \
    --optimizer adadelta --lr 0.1 --final-epochs 8 \
    --use-lrn --dropout 0.5 --weight-decay 0.003 --grad-clip 0.5 \
    --balance-strategy oversample --oversample-target 100000 \
    --no-augment --no-tta --skip-faza-a \
    --seed 42 \
    --output results/figures/modelC_fullslide_faithful_v20_gc05_seed42.npz
```

Zastavice so navedene izrecno, tudi tiste, ki se ujemajo s privzetimi
vrednostmi skripte -- privzetki namreč ustrezajo posodobljeni različici
modela (optimizator Adam, brez lokalne odzivne normalizacije, z
augmentacijo), ne pa članku zvesti konfiguraciji zgoraj. Brez izrecno
naštetih zastavic ukaz nauči bistveno drugačen model.

## Že natrenirani modeli in rezultati

V `results/figures/` so poleg skript za grafe/tabele iz diplome tudi
**že izračunane napovedi flagship modelov** (`.npz`, verjetnosti na testni
množici) in **natreniran RBF SVM model** (`.joblib`). S temi datotekami
lahko kdorkoli regenerira vse grafe in tabele iz diplome (`results/figures/*.py`),
ne da bi imel dostop do surovih/predobdelanih podatkov ali ponovno
treniral modele. Za dejansko ponovitev treninga (in preverjanje rezultatov
iz zgornje tabele) so seveda potrebni podatki -- glej razdelek "Podatki".

## Vir

Berisha, S., Lotfollahi, M., Jahanipour, J., Gurcan, I., Walsh, M.,
Bhargava, R., Van Nguyen, H., in Mayerich, D. (2019). Deep learning for
FTIR histology: leveraging spatial and spectral features with
convolutional neural networks. *The Analyst*, 144(5), 1642--1653.
https://doi.org/10.1039/C8AN01495G

## Licenca

Izvorna koda je na voljo pod licenco GNU General Public License v3.0 (glej
`LICENSE`).
