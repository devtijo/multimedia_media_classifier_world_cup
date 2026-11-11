# Multimodal Media Classifier - key results

Dataset: 438 clips from 23 World Cups, 5 eras (1930-38: 50, 1950-66: 101, 1970-86: 68, 1990-2006: 87, 2010-26: 132).
Validation: leave-one-World-Cup-out (no clip of a tested video is ever seen in training).

## Era classification

| scenario | model | clip acc [95% CI] | macro-F1 | World-Cup-level acc |
|---|---|---|---|---|
| all eras | Baseline: random guess | 0.16 [0.11-0.22] | 0.13 | 0.17 |
| all eras | Image (logistic) | 0.47 [0.34-0.64] | 0.44 | 0.65 |
| all eras | Audio (logistic) | 0.49 [0.29-0.67] | 0.41 | 0.48 |
| all eras | Audio w/o voice features (logistic) | 0.49 [0.30-0.67] | 0.42 | 0.48 |
| all eras | Early fusion (logistic) | 0.58 [0.39-0.75] | 0.49 | 0.61 |
| all eras | Early fusion (random forest) | 0.53 [0.34-0.71] | 0.44 | 0.57 |
| all eras | Late fusion (avg. probabilities) | 0.56 [0.36-0.76] | 0.49 | 0.61 |
| colour-era World Cups only | Baseline: random guess | 0.29 [0.21-0.38] | 0.26 | 0.47 |
| colour-era World Cups only | Image (logistic) | 0.51 [0.33-0.77] | 0.48 | 0.67 |
| colour-era World Cups only | Audio (logistic) | 0.73 [0.52-0.88] | 0.71 | 0.60 |
| colour-era World Cups only | Audio w/o voice features (logistic) | 0.72 [0.51-0.87] | 0.70 | 0.60 |
| colour-era World Cups only | Early fusion (logistic) | 0.74 [0.50-0.91] | 0.68 | 0.67 |

Permutation test (early fusion): observed 0.575, null mean 0.174, p = 0.005 (200 shuffles).

Year regression: MAE 11.8 years per World Cup (baseline 25.1).

## Features that change most with the era (edition level)

| feature | modality | ε² | Spearman ρ vs year | FDR q |
|---|---|---|---|---|
| voiced_ratio | Audio | 0.98 | +0.70 | 0.00995 |
| colorfulness | Image | 0.72 | +0.85 | 0.0395 |
| green_ratio | Image | 0.65 | +0.80 | 0.0395 |
| saturation | Image | 0.62 | +0.73 | 0.0395 |
| dynamic_range_db | Audio | 0.59 | +0.34 | 0.0395 |
| pillarbox | Image | 0.58 | -0.78 | 0.0395 |
| flatness | Audio | 0.57 | +0.59 | 0.0395 |
| hue_entropy | Image | 0.52 | +0.64 | 0.0455 |
| bandwidth_hz | Audio | 0.52 | +0.69 | 0.0455 |
| mfcc11 | Audio | 0.40 | -0.20 | 0.0923 |

9 features differ between eras at FDR q < 0.05.

## Importance

Accuracy lost when a whole modality is shuffled (held-out World Cups): {'Audio': 0.136, 'Image': 0.257}

Top features (random-forest impurity importance):
- colorfulness (Image): 0.090
- saturation (Image): 0.082
- pillarbox (Image): 0.082
- green_ratio (Image): 0.056
- flatness (Audio): 0.044
- loudness_db (Audio): 0.043
- mfcc7 (Audio): 0.043
- hue_entropy (Image): 0.041

## Degradation test (is the model learning quality instead of history?)

Modern clips (2010-26) were artificially degraded. Median predicted-year shift: 1.0 years; 4% of degraded clips leave their era (vs 5% of originals).

## Text keywords per edition (TF-IDF)

- 2010: asker, fabric, netherlands, yester, surely, cassius, minutes, ramos, won, dutch
- 2014: janeiro, rio, topstay, egwain, lavexy, tight, post, german, messy, germany
- 2018: pogba, grisman, france, bombape, penalty, manjukic, possibly, generation, larise, lead
- 2026: williams, martinez, amy, ferritoris, marino, petro, gorgeous, turned, extra, corner

Share of clips with speech per era: {'1930-38': 0.0, '1950-66': 0.0, '1970-86': 0.0, '1990-2006': 0.0, '2010-26': 0.8}
Commentary language detected by Whisper: {1930: 'en', 1934: 'en', 1938: 'en', 1950: 'en', 1954: 'en', 1958: 'en', 1962: 'en', 1966: 'en', 1970: 'en', 1974: 'en', 1978: 'en', 1982: 'en', 1986: 'en', 1990: 'en', 1994: 'en', 1998: 'en', 2002: 'en', 2006: 'en', 2010: 'en', 2014: 'en', 2018: 'en', 2022: 'en', 2026: 'en'}

## Rule-based labels (chi-square vs era, clip level - indicative only, clips are not independent)

- color_mode: chi2 = 447.87, p = 1.06e-91
- texture: chi2 = 29.04, p = 0.000312
- noise_level: chi2 = 142.69, p = 6.53e-27
- pitch_register: chi2 = 265.01, p = 1e-49
- speech: chi2 = 320.16, p = 2.1e-64

## Figures

- fig01_dataset_overview.png
- fig02_frame_montage.png
- fig03_dominant_palettes.png
- fig04_image_features_by_era.png
- fig05_image_trends.png
- fig06_audio_features_by_era.png
- fig07_audio_spectrum_by_era.png
- fig08_dominant_pitch.png
- fig09_rule_based_labels.png
- fig10b_speech_by_era.png
- fig11_feature_effects.png
- fig12_feature_correlation.png
- fig13_pca.png
- fig14_model_comparison.png
- fig15_confusion_matrix.png
- fig16_year_prediction.png
- fig17_feature_importance.png
- fig18_degradation_test.png
- fig19_permutation_test.png