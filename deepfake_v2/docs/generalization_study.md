# Generalization study: design

## Why

The matched ablation (`../README.md`) found no robust in-domain gain from a feature gate or an FFT branch on a
MobileNetV2 detector: all four variants sit at about 0.97 frame AUC on FaceForensics++ C23. In-domain accuracy
is also the setting where deepfake detectors are known to be most misleading. In Meta's Deepfake Detection
Challenge, the top public-leaderboard score was 82.56%, but the winning model scored 65.18% on the withheld set
(sources disagree on whether the metric is accuracy or average precision). Published cross-dataset averages for
strong research detectors, such as the roughly 0.78 AUC reported in DeepfakeBench, are also far below their
in-domain scores.

The frequency idea is usually motivated by robustness and generalization, not by in-domain accuracy. So this study
puts the question where the idea should matter: does the FFT branch help on a manipulation the model has never
seen, or on frames that have been compressed, blurred or shrunk?

## Question

For a detector trained without one manipulation type, how much does it lose on that type, and does the variant with
the FFT branch lose less than the baseline? How quickly does detection fall as test frames are degraded, and does
the FFT branch change that?

## Design

**Folds.** Five manipulation types: Deepfakes, Face2Face, FaceShifter, FaceSwap and NeuralTextures
(DeepFakeDetection stays excluded, as in the clean split). Fold *k* removes type *k* from train and from
validation. Validation must not contain it, or the held-out type would influence which epoch is kept. The type stays
in the test split. Real faces are untouched.

**Split.** The identity-disjoint split from the ablation, seed 0, so every fold, variant and seed is tested on
exactly the same frames, and no test identity appears in training.

**Variants.** `base` against `freq` (baseline plus the FFT branch), the cleanest contrast. The gate is left out
because the ablation showed no effect for it. The design adds `attn` and `hybrid` later if the FFT shows an effect.

**Protocol.** Identical to the ablation: Adam at lr 1e-4, label-smoothed cross-entropy, ReduceLROnPlateau, a
balanced real/fake subset redrawn each epoch, five epochs, best epoch by validation AUC, test read once. Seeds 0 and 1.

**Degradations.** After each run, the best model scores the test frames again after: JPEG quality 70, JPEG quality
40, Gaussian blur of radius 1.5, and a 2x downscale followed by enlargement. They are deterministic and nothing is
trained on them.

## What is measured

- **Unseen AUC:** the held-out type against all real test frames.
- **Seen AUC:** every other type against the same real frames.
- **Gap:** seen minus unseen. A smaller gap means better generalization.
- **Robustness:** seen AUC under each degradation.

All of them are averaged over seeds within a fold, then over folds. Intervals come from a stratified bootstrap over
test videos, resampled jointly for every run, so a difference between two variants uses the same resampled videos
on both sides. Because the report makes many comparisons, it states how likely a spurious "gain" is.

## Decisions and why

- **AUC, not accuracy.** The test set is 83% fake and the held-out type is a fifth of the fakes. AUC does not depend
  on the threshold or on the class split.
- **Two seeds, not three.** Twenty runs at about 15 minutes each is already about five hours on a Kaggle T4. Two
  seeds give a rough noise estimate; the report says so.
- **Degradations at test time only.** This measures robustness of the trained model. Training on degraded frames
  would be a different experiment.

## What this cannot show

- Another dataset (Celeb-DF, DFDC). "Unseen" means one manipulation method on the same kind of source footage, not
  new people, cameras or compression pipelines.
- That synthetic degradations match real social-media processing.
- Anything about a better frequency branch. The FFT branch is the original design, ending in global average
  pooling, which discards frequency position. A null result here says this design does not help.

## Possible outcomes, all reportable

The FFT branch helps generalization; it hurts it; or the difference cannot be resolved with this much data. The
last outcome is the likeliest given the ablation, and it would itself be a result: it would say that this
frequency design does not buy generalization at this scale.

## Cost

20 runs with the defaults. The ablation took about 13 minutes per run on a T4. The four degraded passes should add
a couple of minutes per run (not measured). A first look needs two runs: `--holdouts NeuralTextures --seeds 0`.
