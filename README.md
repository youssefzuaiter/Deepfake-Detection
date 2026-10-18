# Hybrid Deepfake Detection: Spatial-Frequency CNN with Attention

A binary real/fake classifier for face images, trained on FaceForensics++ C23. The project compares three models:

1. **Baseline:** ImageNet-pretrained MobileNetV2 with a 2-way head.
2. **Attention CNN:** the same backbone with a learned 1×1 convolution + sigmoid gate on its feature map.
3. **Hybrid Spatial-Frequency CNN:** the attention model plus a small CNN on the log-scaled FFT magnitude spectrum, with a dense fusion head.

Course project for *Deep Learning*, by Youssef Sufian Zuaiter.

## Results

These are the numbers the notebooks actually produce, on the dataset's own validation and test splits.

| Model | Val acc | Test acc | Test AUC-ROC | Balanced test acc |
|---|---:|---:|---:|---:|
| Baseline MobileNetV2 | 87.52% | 88.59% | not computed | 86.80% (6,000-image subset) |
| Attention CNN | 90.78% | 91.03% | not computed | 85.10% (6,000-image subset) |
| Hybrid Spatial-Frequency CNN | 92.47% | 92.84% | 0.9748 | ≈92.2% (full test set, from confusion matrix) |

The test set has 26,276 images, of which 3,944 are real and 22,332 are fake. The "Balanced test acc" column uses a different test subset for the first two rows, so compare it with care.

**Hybrid per-class results** (test set, `sklearn.classification_report`):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Real (0) | 0.70 | 0.91 | 0.79 | 3,944 |
| Fake (1) | 0.98 | 0.93 | 0.96 | 22,332 |
| **Accuracy** | | | **0.93** | 26,276 |

Confusion matrix for the hybrid model on the test set:

| | Predicted Real | Predicted Fake |
|---|---:|---:|
| **Actual Real** | 3,596 | 348 |
| **Actual Fake** | 1,533 | 20,799 |

Real precision is the weakest figure. The test set is about 85% fake, so false positives on real faces weigh heavily.

### The ablation is not controlled

The three models were not trained under the same conditions:

| | Baseline | Attention CNN | Hybrid |
|---|---|---|---|
| Notebook | `DEEPLEARNING.ipynb` (not in this repo) | `DEEPLEARNING.ipynb` | `Hybrid Deepfake.ipynb` |
| Training images | 10,000 (5,000 per class) | 10,000 (5,000 per class) | 36,782 (18,391 per class) |
| Epochs | 3 | 3 | 5 |
| Loss | cross-entropy | cross-entropy | cross-entropy, label smoothing 0.1 |
| LR schedule | none | none | ReduceLROnPlateau |
| Optimizer | Adam, lr 1e-4 | Adam, lr 1e-4 | Adam, lr 1e-4 |

Because of these differences, the gains reported in the proposal (+2.44 points for attention, +4.75 for the hybrid) cannot be attributed to the architecture alone. On the measured test set, the differences are +2.44 points for attention over baseline, +4.25 points for the hybrid over baseline, and +1.81 points for the hybrid over attention.

On the balanced test subset, the attention model scores 1.70 points below the baseline (85.10% against 86.80%). The attention gain does not hold up under balanced evaluation. The hybrid was not evaluated on that subset, so its balanced accuracy above is computed from its confusion matrix on the full test set.

### Discrepancies between the documents and the notebooks

| Claim | Where | What the notebook measures |
|---|---|---|
| Hybrid test accuracy 93.34%, val 92.83% | Proposal; markdown and chart in hybrid notebook | Test 92.84%, val 92.47% (cell 31). The ablation chart uses hard-coded values. |
| Real precision 0.74 | Project PDF; markdown in hybrid notebook | 0.70 (classification report) |
| Fake precision 0.97, recall 0.95 | Summary cell 40 in hybrid notebook | 0.98 and 0.93 (classification report) |
| FFT branch adds +2.31% over attention | Summary cell 40 in hybrid notebook | +1.81 points using measured numbers (92.84 − 91.03) |
| Balanced subset redrawn each epoch | Project PDF | Drawn once with seed 42 (cell 13) |
| Grad-CAM for the hybrid model | Project PDF | Grad-CAM code exists only for the attention model, in `DEEPLEARNING.ipynb` |

Where the documents disagree with the notebook outputs, treat the notebook outputs as the reference.

## Architecture

Input: 224×224 RGB face crop, ImageNet mean/std normalization.

**Spatial branch**
- MobileNetV2 `features` (ImageNet weights, all layers fine-tuned), producing a 1280×7×7 feature map.
- Attention gate: 1×1 convolution (1280→1280) followed by a sigmoid. The output is multiplied element-wise with the feature map.
- Global average pooling gives a 1280-d vector.

The gate is computed per position and per channel. It is a learned spatial-channel gate rather than strictly channel-wise attention.

**Frequency branch** (hybrid only)
- Average the RGB channels to grayscale.
- 2D FFT, `fftshift`, magnitude, `log1p`, then per-image min-max normalization to a 1×224×224 tensor.
- Three conv blocks, channels 1→16→32→64, each Conv3×3 → BatchNorm → ReLU. Max-pooling follows the first two blocks.
- Global average pooling gives a 64-d vector.

**Fusion and head** (hybrid)
- Concatenate: 1280 + 64 = 1344.
- Linear(1344→256) → ReLU → Dropout(0.4) → Linear(256→64) → ReLU → Dropout(0.3) → Linear(64→2).

**Baseline:** MobileNetV2 with its final classifier replaced by Linear(1280→2).

## Data

- **Source:** Kaggle dataset `gradientvoyager/faceforensics-c23-extracted-faces-100k` (about 2.46 GB). The Kaggle CLI reports its license as CC-BY-NC-SA-4.0. The data is not included in this repository.
- **Classes:** seven folders: `DeepFakeDetection`, `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures`, `Real`. Binary labels: `Real` = 0, all others = 1.
- **Splits:** `dataset_processed_split/{train,val,test}` contains 128,477, 27,493, and 26,276 images. The dataset also includes `dataset_manifest.csv`, which lists every image with its source ID, frame index, label, manipulation type, and split.

| Class | Train | Val | Test |
|---|---:|---:|---:|
| Real | 18,391 | 4,109 | 3,944 |
| Deepfakes | 18,085 | 4,017 | 3,875 |
| Face2Face | 18,365 | 4,105 | 3,954 |
| FaceShifter | 17,481 | 3,889 | 3,715 |
| FaceSwap | 18,885 | 4,209 | 4,044 |
| NeuralTextures | 17,390 | 3,879 | 3,744 |
| DeepFakeDetection | 19,880 | 3,285 | 3,000 |

**Filenames.** Each filename starts with the source video ID, which matches `source_id` in the manifest for every image. Manipulated images have a second numeric ID. I take this to be the target video, but the dataset does not document it.

**Effective sample size.** The images come from few source videos. The training split has 674 source IDs, averaging about 190 frames each, and the validation and test splits have 149 and 147. The number of independent samples is much smaller than the image count.

**Balanced subset.** The baseline and attention notebook keeps up to 5,000 real and 5,000 fake training images. The hybrid notebook keeps all 18,391 real images and draws 18,391 fakes with `random.seed(42)`. In both cases the subset is drawn once and is not resampled per epoch.

### Possible leakage between splits

Source IDs never overlap across the three splits. However, the target side of manipulated videos does overlap:

- 106 of the 147 test source IDs appear as the target of a manipulated video in the training split.
- 100 of the 149 validation source IDs appear as targets in the training split.

A manipulated video's target is the footage its source face is pasted onto, so the training set may contain footage of people who also appear in the test reals. That is a plausible route to inflated test accuracy. I have not measured its effect.

## Training

### Hybrid (`Hybrid Deepfake.ipynb`)

| Setting | Value |
|---|---|
| Epochs | 5 |
| Batch size | 32 |
| Optimizer | Adam, lr = 1e-4 |
| Loss | CrossEntropyLoss, label smoothing 0.1 |
| Scheduler | ReduceLROnPlateau on validation accuracy (mode = max, patience = 1, factor = 0.5) |
| Train augmentation | Resize 224, RandomHorizontalFlip, RandomRotation ±10°, ColorJitter (0.2 brightness/contrast/saturation) |
| Val/test transforms | Resize 224 and ImageNet normalization only |
| Checkpoint | Saved whenever validation accuracy improves |

Per-epoch results from the saved output:

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 0.4642 | 81.81% | 0.3887 | 87.98% |
| 2 | 0.3462 | 91.20% | 0.4167 | 87.31% |
| 3 | 0.3080 | 93.83% | 0.4207 | 87.02% |
| 4 | 0.2763 | 95.77% | 0.3499 | 91.58% |
| 5 | 0.2671 | 96.32% | 0.3359 | 92.47% |

Validation accuracy was still rising at epoch 5, so longer training may help. That is untested.

### Baseline and attention (`DEEPLEARNING.ipynb`)

Both models use Adam (lr 1e-4), plain cross-entropy, batch size 32, and 3 epochs on the 10,000-image balanced subset. Training accuracy after epoch 3 was 96.18% (baseline) and 96.50% (attention).

## Repository contents

- `Hybrid Deepfake.ipynb`: the hybrid model pipeline, covering data, balanced sampling, FFT transform, model, training, evaluation, plots, and weight saving. The same file appears in the project folder as `Deepfake_Detection_Project.ipynb`.
- `Proposal.pdf`: the project proposal.
- `Deepfake_Detection_Project_README.pdf`: an earlier design summary. It contains the discrepancies listed above.

Kept in the project folder but **not** in this repository:
- `DEEPLEARNING.ipynb`: the baseline and attention notebook. It includes the Grad-CAM visualizations for the attention model.
- The dataset (`DATASET.zip`, about 2.6 GB, and the extracted `dataset_processed_split/`).

## Setup

**Environment.** Run on a CUDA GPU. The project PDF reports a Tesla T4 on Google Colab. Both notebooks fall back to CPU if CUDA is unavailable, but training will be very slow.

**Install:**

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy pandas kaggle grad-cam
```

The baseline notebook installs `grad-cam` itself in cell 32.

**Kaggle credentials.** Set your credentials as environment variables or in Colab Secrets. Do not put a key in a notebook, and do not commit one.

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

**Do not run cell 4 of `Hybrid Deepfake.ipynb` as written.** It contains a hard-coded Kaggle key. Replace that cell with the environment-variable approach above.

**Data paths.** Both notebooks expect the Colab layout `/content/dataset/dataset_processed_split/{train,val,test}`, set as `data_dir`. For a local run, change that path to where you unzip the dataset.

## Running it

**Hybrid model** (`Hybrid Deepfake.ipynb`):
1. Set the Kaggle credentials as described above.
2. Run the download and unzip cells (6–7), data and loader cells (9–17), model cells (19–20), training cells (22–25), and evaluation cells (31, 36–38), then save the weights (cell 42).

Outputs: `best_hybrid_model.pth`, `hybrid_deepfake_model.pth`, `hybrid_training_curves.png`, `roc_curve.png`, `ablation_study.png`. The ablation chart uses the hard-coded numbers described above.

**Baseline and attention** (`DEEPLEARNING.ipynb`):
1. Set the Kaggle credentials as described above. Cell 2 contains placeholders only.
2. Run the cells in order through cell 37 for the balanced evaluation. Cells 31–34 produce the Grad-CAM figures.

Outputs: `training_accuracy_comparison.png`, `training_loss_comparison.png`, `roc_attention.png`, `sample_predictions.png`.

To load the hybrid weights for inference, run the imports (cell 2) and the model definition (cell 19), then construct the model and load the state dict. The model class is defined in the notebook, not in a `.py` module.

## Limitations

- **Confounded ablation.** The three models differ in training data size, epochs, loss, and scheduler (see above). The attention gain also disappears on the balanced test subset.
- **Possible split leakage.** Source videos do not overlap, but many test sources appear as targets in training (see Data).
- **Few independent samples.** The test set draws on 147 source videos.
- **Single distribution.** Training and testing use FaceForensics++ C23 only. Generalization to DFDC or Celeb-DF is untested.
- **Compressed source data.** C23 is compressed video. A detector can partly learn compression artifacts, and performance on other compression levels is untested.
- **Class precision.** Real precision is 0.70 on the imbalanced test set, so many real faces are flagged as fake.
- **Short training.** Three and five epochs, with validation accuracy still rising at the end.
- **No variance estimate.** Each model comes from a single run. Only the data subsets are seeded, so there are no error bars.
- **Documents out of sync.** See the discrepancy table. The hybrid's headline figure in the proposal (93.34%) is not reproduced.
- **Weights not included.** The trained model files are not in this repository.
- **Not a security tool.** This is a research prototype. Do not use it for real-world authenticity decisions.

## License

No license file is included with the code, so the code's license has not been specified. The dataset is licensed CC-BY-NC-SA-4.0 by its Kaggle listing.
