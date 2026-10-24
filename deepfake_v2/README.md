# Deepfake detection v2: honest evaluation and a matched ablation

This folder replaces the ablation part of the original notebooks. The model idea is unchanged
(MobileNetV2 + optional gate + optional FFT branch). What changes is how it is trained and measured,
so that every number can be defended.

## What it fixes

| Problem in the notebooks | What v2 does |
|---|---|
| Baseline, attention and hybrid were trained with different data, epochs, loss and scheduler, so gains could not be attributed | One protocol for all four variants (gate off/on x FFT off/on), 3 seeds each |
| Hard-coded result tables (93.34%) that the code never produced | Every number is computed from saved predictions by `report.py` |
| Accuracy on a test set that is 85% fake | AUC, balanced accuracy, EER, TPR at 1% FPR, per-manipulation results, and the always-"fake" baseline printed next to accuracy |
| 229 of 295 test identities also appear in training (fake videos were assigned to a split by only one of their two source ids) | Identity-disjoint re-split, on by default (`--split clean`): every original video and actor lives in exactly one split, and this is audited at the start of every run |
| Frames treated as independent | Video-level decisions and 95% intervals from resampling test **videos** |
| 83% of training fakes discarded forever | A new balanced real/fake subset is drawn every epoch |
| Best checkpoint saved but never loaded; threshold implicitly 0.5 | Best epoch chosen on validation AUC and reloaded; test read once; one extra metric uses a validation-chosen threshold |
| One run, one seed | Seeds set for python, numpy and torch; mean and std over seeds |

## Layout

```
dfv2/metrics.py       pure numpy metrics, weighted so the bootstrap can resample videos
dfv2/manifest.py      manifest -> frames with a video id; balanced per-epoch sampling
dfv2/splits.py        identity-disjoint re-split and the leakage audit
dfv2/make_split.py    prints the audit of the original split next to the clean one
dfv2/models.py        one model class for all four variants; FFT matches the original code exactly
dfv2/dataset.py       torch dataset and the original augmentations
dfv2/train.py         one (variant, seed) run -> predictions CSV, history JSON, weights
dfv2/run_ablation.py  loops variants x seeds, skips runs that already finished
dfv2/report.py        predictions -> report.md, results.json, figures (no torch needed)
tests/                pytest suite (metrics checked against scikit-learn, report checked on synthetic data)
build_notebook.py     builds the Kaggle and Colab notebooks from dfv2/ so they never drift apart
```

## Run on Kaggle (recommended) or Colab

**On Kaggle:** [`../Deepfake_Detection_v2_kaggle.ipynb`](../Deepfake_Detection_v2_kaggle.ipynb). Import it (New notebook >
File > Import notebook), set Accelerator to GPU T4 x2 and Internet to On, add the FaceForensics++ C23 dataset with
**+ Add Input**, and run the cells top to bottom. The dataset is read in place: no download and no keys.

**On Colab:** run `python build_notebook.py`, which writes `Deepfake_Detection_v2.ipynb` (Colab) and
`Deepfake_Detection_v2_kaggle.ipynb` (Kaggle) next to this folder. Upload the Colab one, pick a GPU runtime, add
`KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab Secrets, and run the cells top to bottom. The Colab path has not been run
end to end; the Kaggle notebook has.

Each notebook contains the whole package (each module is a `%%writefile` cell), the split check, a quick GPU test, the
experiment and the report. They are generated from this folder, so after changing any file in `dfv2/`, rebuild them
rather than editing the notebooks.

Or, without the notebook: upload this folder (or copy it to Drive), then:

```python
from google.colab import drive, userdata
import os
drive.mount("/content/drive")
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")   # keep credentials in Colab Secrets
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")             # never paste them into a notebook cell
```

```bash
%cd /content/deepfake_v2
!pip install -q -r requirements.txt kaggle
!kaggle datasets download -d gradientvoyager/faceforensics-c23-extracted-faces-100k -p /content
!unzip -q /content/faceforensics-c23-extracted-faces-100k.zip -d /content/dataset
!python -m dfv2.make_split --data-dir /content/dataset/dataset_processed_split   # prints the leakage audit, trains nothing
!python -m dfv2.run_ablation --data-dir /content/dataset/dataset_processed_split \
        --out-dir /content/drive/MyDrive/deepfake_runs --seeds 0
!python -m dfv2.report --runs-dir /content/drive/MyDrive/deepfake_runs
```

Start with `--seeds 0` (four runs) to check everything works, then run again with `--seeds 0 1 2`.
Finished runs are skipped, so a disconnected session can resume with the same command.

Measured on a Kaggle T4 (the full run in `../Deepfake_Detection_v2_full_run.ipynb`): about 12 to 15 minutes per run
once the dataset is cached, and about 4 hours for all 12 runs. The first run was slower (about 1 h 25 min to 1 h 50 min
in two sessions), probably the one-time read of the dataset. The CUDA mixed-precision path ran there without errors. If
it errors elsewhere, add `--no-amp`.

## Run locally

```bash
python -m pytest -q
# tiny end-to-end check on real data, random weights, a few videos
python -m dfv2.run_ablation --data-dir ../dataset_processed_split --out-dir /tmp/smoke \
        --max-videos 6 --epochs 1 --no-pretrained --device cpu --num-workers 0 --seeds 0
python -m dfv2.report --runs-dir /tmp/smoke --n-boot 50
```

## The clean split

Measured on your manifest (`python -m dfv2.make_split`):

| | Original Kaggle split | Clean split (seed 0) |
|---|---|---|
| Frames train / val / test | 128,477 / 27,493 / 26,276 | 109,304 / 23,373 / 23,404 |
| Identities shared between splits | train/val 218, train/test 229, val/test 65 | 0, 0, 0 |
| Test identities also seen in training | 229 of 295 | 0 of 142 |
| Fake share of test frames (always-"fake" accuracy) | 85.0% | 83.1% |

How it works: fake video "A_B" ties original videos A and B together, so every video, actor and fake linked by those
ids is kept in one split. Without DeepFakeDetection the ids form 483 independent pairs, which are dealt out 70/15/15
by frame count. The seed only changes which pairs land where; seeds 0, 1 and 2 all give zero shared identities.
Image files do not move, and the split is recomputed from the manifest each time, so nothing extra needs saving.

Decisions I made that you may want to change:

- **DeepFakeDetection is dropped** (26,165 fake frames, 17% of all fakes). It has no real frames from its own
  studio domain, so a model can score on it by recognising the scenes. `--keep-dfd` retains it with its actors also
  kept apart (zero shared identities, checked), at the cost of that shortcut.
- **The test set shrinks slightly** (23,404 frames from 818 videos) and is a different set from the original test
  set, so numbers are not comparable with the earlier notebooks. Compare variants with each other.
- **`--split original`** reproduces the old assignment. Use a different `--out-dir`, because a run folder refuses to
  mix runs made with different settings.

## The shared training protocol

| Setting | Value (all four variants) |
|---|---|
| Backbone | ImageNet-pretrained MobileNetV2, fine-tuned end to end |
| Dense head | 3 layers, 256 and 64 units, dropout 0.4 and 0.3 (input width 1280, or 1344 with FFT) |
| Loss, optimiser | Cross-entropy with label smoothing 0.1; Adam, lr 1e-4 |
| Scheduler | ReduceLROnPlateau on validation AUC (patience 1, factor 0.5) |
| Training data per epoch | All 18,391 real train frames + an equal number of fakes, redrawn every epoch |
| Split | Identity-disjoint, 70/15/15 by frames, DeepFakeDetection dropped |
| Epochs, batch | 5, 32 |
| Model selection | Best validation AUC; test split read once afterwards |

The baseline uses the same dense head as the others, so it is not the original single-layer baseline.

## What this does not fix yet

- **One split draw.** Results come from a single identity-disjoint split (seed 0). Repeating with `--split-seed 1` and `2` would show how much the split itself matters.
- **The frequency branch design.** It is kept as in the original so the ablation tests that design. It pools away frequency positions and reads already-resized JPEG crops.
- **Only FaceForensics++ C23.** No cross-dataset test, no compression-robustness test.
