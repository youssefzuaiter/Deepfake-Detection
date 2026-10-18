# Hybrid Deepfake Detection: Spatial-Frequency CNN with Attention

A binary real/fake classifier for face images, trained on FaceForensics++ C23. The model combines an ImageNet-pretrained MobileNetV2 with a learned attention gate on its spatial features, and a small CNN on the log-scaled FFT magnitude spectrum. The two feature vectors are concatenated and classified by a dense head.

Course project for *Deep Learning*, by Youssef Sufian Zuaiter.

## Results

Measured by the notebook `Hybrid Deepfake.ipynb` on the dataset's own validation and test splits:

| Metric | Value |
|---|---|
| Validation accuracy (epoch 5, best checkpoint) | 92.47% |
| Test accuracy | 92.84% |
| Test AUC-ROC | 0.9748 |

Per-class test metrics (`sklearn.classification_report`):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Real (0) | 0.70 | 0.91 | 0.79 | 3,944 |
| Fake (1) | 0.98 | 0.93 | 0.96 | 22,332 |
| **Accuracy** | | | **0.93** | 26,276 |

The test set is about 85% fake, so false positives on real faces weigh heavily in the Real precision. That is the weakest number here.

**Ablation numbers are not reproduced here.** The project proposal reports a three-model ablation: baseline MobileNetV2 at 88.59% test accuracy, an attention-only CNN at 91.03%, and the hybrid at 93.34%. Neither the baseline nor the attention-only model is in this repository. The hybrid's measured test accuracy in the notebook is 92.84%, not 93.34%. The ablation table and `ablation_study.png` in the notebook are built from hard-coded numbers, not from runs in the notebook.

## Architecture

Input: 224×224 RGB face crop, ImageNet mean/std normalization.

**Spatial branch**
- MobileNetV2 `features` (ImageNet weights, all layers fine-tuned), producing a 1280×7×7 feature map.
- Attention gate: 1×1 convolution (1280→1280) followed by a sigmoid. The output is multiplied element-wise with the feature map.
- Global average pooling gives a 1280-d vector.

The gate is computed per position and per channel, so it is a learned spatial-channel gate rather than strictly channel-wise attention.

**Frequency branch**
- Convert RGB to grayscale by averaging channels.
- 2D FFT, `fftshift`, magnitude, `log1p`, then per-image min-max normalization to a 1×224×224 tensor.
- Three conv blocks (Conv3×3 → BatchNorm → ReLU), channels 1→16→32→64, with max-pooling after the first two blocks.
- Global average pooling gives a 64-d vector.

**Fusion and head**
- Concatenate: 1280 + 64 = 1344.
- Linear(1344→256) → ReLU → Dropout(0.4) → Linear(256→64) → ReLU → Dropout(0.3) → Linear(64→2).

## Data

- **Source:** Kaggle dataset `gradientvoyager/faceforensics-c23-extracted-faces-100k` (about 2.46 GB). The Kaggle CLI reports its license as CC-BY-NC-SA-4.0. The data is not included in this repository.
- **Classes:** seven folders, `DeepFakeDetection`, `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures`, and `Real`. Binary labels: `Real` = 0, every other class = 1.
- **Splits** (from `dataset_processed_split/{train,val,test}`): train 128,477, validation 27,493, test 26,276.
- **Training balance:** the train folder has 18,391 real and 110,086 fake images. The notebook keeps all real images and draws 18,391 fakes with `random.seed(42)`. This gives 36,782 training images. The subset is drawn once; it is not resampled each epoch.

**Caveat on splits:** the notebook uses the split as provided. I have not checked whether the splits are disjoint at the source-video level. If frames from the same video appear in more than one split, the reported accuracy would be optimistic.

## Training

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

Per-epoch results from the saved notebook output:

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 0.4642 | 81.81% | 0.3887 | 87.98% |
| 2 | 0.3462 | 91.20% | 0.4167 | 87.31% |
| 3 | 0.3080 | 93.83% | 0.4207 | 87.02% |
| 4 | 0.2763 | 95.77% | 0.3499 | 91.58% |
| 5 | 0.2671 | 96.32% | 0.3359 | 92.47% |

Validation accuracy was still rising at epoch 5. Five epochs is a short run, and longer training has not been tested here.

## Repository contents

- `Hybrid Deepfake.ipynb`: the full pipeline. It covers data loading, balanced sampling, the FFT transform, the model, training, evaluation, plots, and weight saving.
- `Proposal.pdf`: the project proposal.
- `Deepfake_Detection_Project_README.pdf`: an earlier design summary. Where it and the notebook disagree, the notebook is the reference. For example, the PDF says the balanced subset is redrawn each epoch, and the notebook draws it once. The PDF also describes Grad-CAM results, but the notebook contains no Grad-CAM code.

## Setup

**Environment.** Run on a CUDA GPU. The project PDF reports a Tesla T4 on Google Colab. The notebook falls back to CPU if CUDA is unavailable, but training will be very slow.

**Install:**

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy pandas kaggle
```

`pytorch-grad-cam` is listed in the project PDF but is not imported by the notebook.

**Kaggle credentials.** Set your credentials as environment variables or in Colab Secrets. Do not put a key in the notebook, and do not commit one.

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

**Data paths.** The notebook expects the Colab layout, `/content/dataset/dataset_processed_split/{train,val,test}`, set in cell 9 as `data_dir`. For a local run, change that path to match where you unzip the dataset.

## Running it

1. Open `Hybrid Deepfake.ipynb` in Colab with a GPU runtime, or in local Jupyter with CUDA.
2. Set the Kaggle credentials as described above.
3. Run the cells in order:
   - Download and unzip the dataset (cells 6–7).
   - Build the datasets and loaders (cells 9–17).
   - Define and build the model (cells 19–20).
   - Train (cells 22–25).
   - Evaluate and plot (cells 31, 36–38).
   - Save the weights (cell 42).

**Outputs:**
- `best_hybrid_model.pth`: the best-validation checkpoint.
- `hybrid_deepfake_model.pth`: the final weights.
- `hybrid_training_curves.png`, `roc_curve.png`, `ablation_study.png`.

To load saved weights for inference, run the imports (cell 2) and the model definition (cell 19), then build the model and load the state dict. The class lives in the notebook, not in a `.py` module.

## Limitations

- **Single distribution.** Training and testing both use FaceForensics++ C23. Generalization to other datasets such as DFDC or Celeb-DF has not been tested.
- **Compressed source data.** C23 is compressed video. Performance on uncompressed or differently compressed media is untested, and a detector can partly learn compression artifacts.
- **Class precision.** Real-class precision is 0.70 on the imbalanced test set, so many real faces are flagged as fake.
- **Short training.** Five epochs, with validation accuracy still rising at the end.
- **No variance estimate.** The reported model comes from a single run. Only the data subset is seeded, so there are no error bars.
- **Split methodology.** Source-video disjointness of the splits is unverified (see Data).
- **Missing code.** The baseline and attention-only models from the ablation are not in this repository. The Grad-CAM results described in the PDF are not reproduced by any code here.
- **Not a security tool.** This is a research prototype. It should not be used to make real-world authenticity decisions.

## License

No license file is included with the code, so the code's license has not been specified. The dataset is licensed CC-BY-NC-SA-4.0 by its Kaggle listing.
