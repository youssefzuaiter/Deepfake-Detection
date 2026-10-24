# Matched 2×2 ablation: report

Variants: Baseline (3 seeds), + Gate (3 seeds), + FFT (3 seeds), Gate + FFT (3 seeds).
Every run shares one protocol: same data, same epochs, same loss, same scheduler, same dense head. Only the component under test changes.

**Test set:** 23,404 frames from 818 videos (139 real, 679 fake); 83.1% of frames are fake.
Always answering "fake" would score **83.11%** accuracy per frame and 83.01% per video. Accuracy numbers only mean something relative to that.

## What the data supports

Differences below are in AUC (0 to 1 scale), with a 95% interval from resampling test videos. "Not resolved" means the test set is too small to tell the effect from zero, not that the effect is zero.

- **+ Gate minus Baseline**, frame auc: -0.0018 [-0.0043, 0.0004]: not resolved (interval includes 0).
- **+ Gate minus Baseline**, video auc: -0.0000 [-0.0019, 0.0019]: not resolved (interval includes 0).
- **+ FFT minus Baseline**, frame auc: -0.0019 [-0.0044, 0.0006]: not resolved (interval includes 0).
- **+ FFT minus Baseline**, video auc: -0.0009 [-0.0030, 0.0011]: not resolved (interval includes 0).
- **Gate + FFT minus Baseline**, frame auc: 0.0010 [-0.0016, 0.0041]: not resolved (interval includes 0).
- **Gate + FFT minus Baseline**, video auc: 0.0027 [0.0001, 0.0055]: gain.
- **Gate main effect**, frame auc: 0.0005 [-0.0012, 0.0027]: not resolved (interval includes 0).
- **Gate main effect**, video auc: 0.0018 [0.0002, 0.0036]: gain.
- **FFT main effect**, frame auc: 0.0004 [-0.0014, 0.0024]: not resolved (interval includes 0).
- **FFT main effect**, video auc: 0.0009 [-0.0006, 0.0025]: not resolved (interval includes 0).
- **Interaction (Gate x FFT)**, frame auc: 0.0048 [0.0011, 0.0087]: gain.
- **Interaction (Gate x FFT)**, video auc: 0.0036 [0.0011, 0.0066]: gain.

## How to read the numbers

- **AUC**: probability that a random fake frame scores higher than a random real one. Ignores the threshold and the real/fake class split.
- **Balanced accuracy**: mean of recall on fakes and recall on reals. Not affected by the class split.
- **EER**: error rate at the threshold where wrongly-flagged reals equal missed fakes. Lower is better.
- **TPR @1% FPR**: share of fakes caught while flagging at most 1% of real faces.
- **Real precision**: of faces called real, the share that are. It falls when fakes are common even if the model treats both classes equally, so it is not a measure of bias toward "fake".
- **Bal acc @val-EER thr**: balanced accuracy at a threshold chosen on validation data, not on test.
- **Video-level**: the mean P(fake) over a video's frames, one decision per video.
- **± values** are the standard deviation across training seeds. **Intervals** are test-set uncertainty. They answer different questions.

## 1. Frame-level metrics (mean ± std over seeds; percentages except AUC)

| Model | Seeds | Accuracy @0.5 | Balanced acc @0.5 | AUC | EER | TPR @1% FPR | Real recall | Fake recall | Real precision | Bal acc @val-EER thr |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 3 | 91.74 ± 1.03 | 91.55 ± 0.55 | 0.9714 ± 0.0011 | 8.15 ± 0.28 | 74.03 ± 1.21 | 91.28 ± 2.92 | 91.83 ± 1.83 | 69.66 ± 3.99 | 91.84 ± 0.32 |
| + Gate | 3 | 92.60 ± 0.57 | 91.48 ± 0.38 | 0.9695 ± 0.0006 | 8.21 ± 0.35 | 68.91 ± 4.17 | 89.79 ± 1.13 | 93.17 ± 0.84 | 72.82 ± 2.22 | 91.77 ± 0.34 |
| + FFT | 3 | 91.69 ± 1.36 | 91.72 ± 0.32 | 0.9694 ± 0.0040 | 8.26 ± 0.37 | 70.27 ± 12.38 | 91.75 ± 1.80 | 91.68 ± 1.97 | 69.40 ± 4.54 | 91.81 ± 0.45 |
| Gate + FFT | 3 | 90.14 ± 2.70 | 91.74 ± 0.93 | 0.9724 ± 0.0022 | 7.96 ± 0.62 | 75.49 ± 3.91 | 94.15 ± 2.59 | 89.33 ± 3.70 | 64.86 ± 7.24 | 92.04 ± 0.61 |

## 2. Video-level metrics (mean ± std over seeds)

| Model | Seeds | Accuracy @0.5 | Balanced acc @0.5 | AUC | EER |
|---|---|---|---|---|---|
| Baseline | 3 | 94.78 ± 0.98 | 95.33 ± 0.40 | 0.9862 ± 0.0010 | 4.64 ± 0.37 |
| + Gate | 3 | 94.95 ± 0.51 | 93.90 ± 0.51 | 0.9861 ± 0.0010 | 5.30 ± 0.26 |
| + FFT | 3 | 94.50 ± 1.17 | 94.68 ± 0.88 | 0.9853 ± 0.0022 | 5.30 ± 0.97 |
| Gate + FFT | 3 | 93.89 ± 1.62 | 95.46 ± 0.47 | 0.9889 ± 0.0010 | 4.79 ± 0.41 |

## 3. Test-set uncertainty (seed-averaged value and 95% interval over resampled videos)

| Model | Frame AUC | Frame balanced acc | Video AUC | Video balanced acc |
|---|---|---|---|---|
| Baseline | 0.9714 [0.9630, 0.9791] | 91.55 [90.13, 92.97] | 0.9862 [0.9796, 0.9922] | 95.33 [93.85, 96.78] |
| + Gate | 0.9695 [0.9606, 0.9776] | 91.48 [89.85, 93.09] | 0.9861 [0.9793, 0.9921] | 93.90 [91.93, 95.86] |
| + FFT | 0.9694 [0.9601, 0.9774] | 91.72 [90.24, 93.15] | 0.9853 [0.9781, 0.9914] | 94.68 [92.91, 96.38] |
| Gate + FFT | 0.9724 [0.9641, 0.9795] | 91.74 [90.51, 92.95] | 0.9889 [0.9834, 0.9937] | 95.46 [94.03, 96.70] |

## 4. Component effects (same resampled videos on both sides)

Differences are in AUC units (0 to 1) for the AUC columns and in percentage points for the balanced-accuracy columns. The gate and FFT main effects average over the other component; the interaction asks whether the two help more (or less) together than separately.

| Comparison | Frame AUC | Frame balanced acc | Video AUC | Video balanced acc | Verdict on frame AUC |
|---|---|---|---|---|---|
| + Gate minus Baseline | -0.0018 [-0.0043, 0.0004] | -0.08 [-0.61, 0.45] | -0.0000 [-0.0019, 0.0019] | -1.43 [-2.78, -0.30] | not resolved (interval includes 0) |
| + FFT minus Baseline | -0.0019 [-0.0044, 0.0006] | 0.16 [-0.31, 0.63] | -0.0009 [-0.0030, 0.0011] | -0.65 [-1.73, 0.32] | not resolved (interval includes 0) |
| Gate + FFT minus Baseline | 0.0010 [-0.0016, 0.0041] | 0.19 [-0.37, 0.77] | 0.0027 [0.0001, 0.0055] | 0.13 [-0.67, 0.95] | not resolved (interval includes 0) |
| Gate main effect | 0.0005 [-0.0012, 0.0027] | -0.03 [-0.39, 0.38] | 0.0018 [0.0002, 0.0036] | -0.33 [-1.05, 0.38] | not resolved (interval includes 0) |
| FFT main effect | 0.0004 [-0.0014, 0.0024] | 0.21 [-0.18, 0.64] | 0.0009 [-0.0006, 0.0025] | 0.45 [-0.21, 1.16] | not resolved (interval includes 0) |
| Interaction (Gate x FFT) | 0.0048 [0.0011, 0.0087] | 0.10 [-0.79, 1.09] | 0.0036 [0.0011, 0.0066] | 2.20 [0.26, 4.38] | gain |

## 5. By manipulation type

AUC of each fake type against all real test frames (mean ± std over seeds). Frames per type in the test set: Deepfakes 3,867, Face2Face 3,948, FaceShifter 3,779, FaceSwap 4,070, NeuralTextures 3,788, Real 3,952.

| Manipulation | Baseline | + Gate | + FFT | Gate + FFT |
|---|---|---|---|---|
| Deepfakes | 0.9845 ± 0.0049 | 0.9838 ± 0.0004 | 0.9832 ± 0.0017 | 0.9867 ± 0.0011 |
| Face2Face | 0.9746 ± 0.0039 | 0.9710 ± 0.0036 | 0.9730 ± 0.0061 | 0.9751 ± 0.0003 |
| FaceShifter | 0.9851 ± 0.0028 | 0.9841 ± 0.0042 | 0.9836 ± 0.0024 | 0.9807 ± 0.0034 |
| FaceSwap | 0.9772 ± 0.0054 | 0.9731 ± 0.0051 | 0.9753 ± 0.0122 | 0.9809 ± 0.0062 |
| NeuralTextures | 0.9348 ± 0.0061 | 0.9351 ± 0.0053 | 0.9313 ± 0.0032 | 0.9374 ± 0.0020 |

Accuracy at threshold 0.5 per type; the Real row is recall on real faces.

| Type | Baseline | + Gate | + FFT | Gate + FFT |
|---|---|---|---|---|
| Deepfakes | 95.66 ± 1.23 | 96.77 ± 0.73 | 95.51 ± 2.19 | 93.86 ± 2.32 |
| Face2Face | 92.95 ± 1.89 | 93.98 ± 0.81 | 93.25 ± 1.66 | 90.83 ± 3.27 |
| FaceShifter | 96.78 ± 0.19 | 97.11 ± 1.35 | 96.43 ± 1.35 | 92.94 ± 3.13 |
| FaceSwap | 92.54 ± 3.85 | 93.09 ± 0.56 | 92.24 ± 3.11 | 91.58 ± 5.00 |
| NeuralTextures | 81.06 ± 4.17 | 84.82 ± 2.20 | 80.81 ± 4.22 | 77.10 ± 5.42 |
| Real | 91.28 ± 2.92 | 89.79 ± 1.13 | 91.75 ± 1.80 | 94.15 ± 2.59 |

## 6. Training behaviour

9 of 12 runs had their best validation AUC in the final epoch, so training had not clearly converged and longer runs could change the picture. See `learning_curves.png`. The best epoch was chosen on validation AUC and the test set was read once afterwards.

## 7. Caveats that these numbers do not remove

- **Identity-disjoint split.** No original video or actor appears in more than one split, checked when each run started. Frames: train 109,304, val 23,373, test 23,404. Identities found in more than one split: 0.
- **DeepFakeDetection is excluded.** It has fake frames but no real frames from its own studio domain, so a model could score on it by recognising the scenes, and its actors form linked groups too large to split cleanly. Re-run with `--keep-dfd` to include it.
- **FaceForensics++ C23 only.** Nothing here tests other datasets or other compression levels.
- **The frequency branch is the original design.** It pools away the frequency positions and reads 224×224 JPEG crops after resizing and augmentation. A negative or null FFT result here says this branch does not help, not that frequency cues cannot.
- **The baseline here is not the original baseline.** It uses the same three-layer dense head as the other variants, so the comparison isolates the gate and the FFT branch.

## Figures

`roc_frame.png`, `per_manipulation_auc.png`, `learning_curves.png`, and every number above as `results.json`.
