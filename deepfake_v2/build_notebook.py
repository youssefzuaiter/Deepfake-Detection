"""Build the notebooks from the dfv2 package, so they always match the tested code.

    python build_notebook.py                   # both notebooks, written to the project folder
    python build_notebook.py --target kaggle   # only one of them

The two targets run the same package code and differ only in where the data comes from and where results go:

  kaggle  Deepfake_Detection_v2_kaggle.ipynb  reads the dataset attached under /kaggle/input, writes to /kaggle/working
  colab   Deepfake_Detection_v2.ipynb         downloads the dataset from Kaggle, writes to Google Drive

Each module becomes a `%%writefile dfv2/<name>.py` cell. To change the code, edit the package and rebuild;
do not edit the generated code cells by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODULES = [
    ("__init__.py", "Marks `dfv2` as a Python package."),
    ("metrics.py", "Every metric, written to accept per-row weights so the bootstrap can resample whole "
                   "videos. Convention throughout: label 1 = fake, and a score is P(fake)."),
    ("splits.py", "The identity-disjoint split and the leakage audit. Original videos and actors are nodes, each "
                  "fake video links its two ids, and linked groups are never separated."),
    ("manifest.py", "Reads `dataset_manifest.csv`, or the folder layout when there is no CSV. Works out which "
                    "video each frame came from, applies the split scheme, finds the dataset folder under "
                    "`/kaggle/input`, and draws a fresh balanced real/fake subset each epoch."),
    ("models.py", "One model class for all four variants (feature gate off/on x FFT branch off/on). The FFT "
                  "computation matches the original notebook exactly."),
    ("dataset.py", "Torch dataset, the original augmentations, and the per-epoch balanced sampler."),
    ("train.py", "Trains one (variant, seed) run under the shared protocol and saves predictions, history and "
                 "weights. The test split is read once, after the best epoch has been chosen on validation."),
    ("run_ablation.py", "The command-line runner: loops over seeds and variants, skips finished runs, and refuses "
                        "to mix results made with different settings."),
    ("make_split.py", "Prints the leakage audit of the original split next to the clean one."),
    ("report.py", "Turns saved predictions into the report: tables, bootstrap intervals over test videos, "
                  "per-manipulation results and figures. Needs no GPU."),
]

IDEA = """\
## The idea in short

**Question.** Do two additions to a MobileNetV2 detector help: a learned *feature gate* and an *FFT frequency branch*? Alone, or together?

| Variant | Feature gate | FFT branch |
|---|:---:|:---:|
| `base` | no | no |
| `attn` | yes | no |
| `freq` | no | yes |
| `hybrid` | yes | yes |

**Why "matched".** In the original notebooks the three models were trained on different amounts of data, for different numbers of epochs, with different losses. A gain could not be blamed on the component. Here only the component changes: same data, epochs, loss, scheduler and dense head.

**Why the metrics changed.** About 83% of test frames are fake, so always answering "fake" already scores that much. The report puts accuracy next to that baseline and adds metrics that ignore the class split: AUC, balanced accuracy, equal error rate. It also decides per video and puts 95% intervals around results by resampling test *videos*, because frames from one video are nearly copies of each other.

**What the FFT branch is.** It converts each face to grayscale, takes the 2-D Fourier transform, and feeds the log-magnitude spectrum to a small CNN. The hope is that manipulation leaves traces in the frequency content. In this design the branch ends with global average pooling, which throws away *which* frequencies matter, and it reads 224x224 JPEG crops after resizing and augmentation. A null result for the FFT branch here therefore says this design does not help, not that frequency cues are useless.

**Leakage, in one line.** In the Kaggle split a fake video `A_B` is placed by one of its two ids, so the other id often appears in training. The clean split puts every original video and actor in exactly one split."""

TITLE_KAGGLE = """\
# Deepfake detection v2: a matched 2x2 ablation (Kaggle)

Follow-up to the course project (MobileNetV2 + feature gate + FFT frequency branch on FaceForensics++ C23).
The model idea is unchanged. What changes is how it is trained and measured, so every number can be defended.

**What this notebook does**

1. Builds an **identity-disjoint** train/val/test split: no person or source video appears in two splits.
2. Trains four variants under **one identical protocol**, with several seeds each.
3. Writes a report with confidence intervals, per-manipulation results, and a plain statement of what the data does and does not support.

**Before you run it**

1. **Accelerator:** Settings (right panel) > Accelerator > **GPU T4 x2**. Choose T4 rather than P100: the P100 is an older GPU and may not work with the current PyTorch build.
2. **Internet:** Settings > Internet > **On**. It is needed once, to download the ImageNet weights for MobileNetV2.
3. **Dataset:** **+ Add Input** > search `faceforensics-c23-extracted-faces-100k` (gradientvoyager) > add it. The notebook reads it in place, so nothing is downloaded.

Then run the cells top to bottom. Run the quick test before the experiment.

_Generated from the `deepfake_v2/` package by `build_notebook.py`. To change the code, edit the package and rebuild._"""

TITLE_COLAB = """\
# Deepfake detection v2: a matched 2x2 ablation

Follow-up to the course project (MobileNetV2 + feature gate + FFT frequency branch on FaceForensics++ C23).
The model idea is unchanged. What changes is how it is trained and measured, so every number can be defended.

**What this notebook does**

1. Downloads the dataset and builds an **identity-disjoint** train/val/test split (no person or source video appears in two splits).
2. Trains four variants under **one identical protocol**, several seeds each.
3. Writes a report with confidence intervals, per-manipulation results and a plain statement of what the data does and does not support.

**How to use it:** set the runtime to a GPU (Runtime > Change runtime type > T4), then run the cells from top to bottom.
Run the quick test before the long run. The long run saves to Google Drive and can be restarted after a disconnect: finished runs are skipped.

_Generated from the `deepfake_v2/` package by `build_notebook.py`. To change the code, edit the package and rebuild._"""

SETTINGS_KAGGLE_MD = """\
## Settings

Constants you can change, plus two checks: the GPU and internet access."""

RUN_HELPER = """\
def run(*args):
    '''Run `python <args>` with this notebook's own interpreter, show its output, and stop the cell if it fails.

    Shell lines such as `!python ...` hide failures, so every step goes through this function.
    '''
    proc = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"failed with exit code {proc.returncode}: python {' '.join(args)}")"""

SETTINGS_KAGGLE_CODE = """\
import os
import socket
import subprocess
import sys

import torch

""" + RUN_HELPER + """

os.chdir("/kaggle/working")          # outputs and the dfv2 package go here
os.makedirs("dfv2", exist_ok=True)   # the module cells below write into this folder

OUT_DIR = "/kaggle/working/runs"     # kept as this notebook's output
SEEDS = [0]                          # [0] first (4 runs); [0, 1, 2] for the full study
EPOCHS = 5
SPLIT = "clean"                      # identity-disjoint; "original" is Kaggle's assignment (use another OUT_DIR)
EXTRA_FLAGS = ""                     # e.g. "--no-amp" if mixed precision errors in the quick test

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU. Settings > Accelerator > GPU T4 x2. On CPU this would take days.")

try:
    socket.create_connection(("download.pytorch.org", 443), timeout=10).close()
    print("internet: ok")
except OSError:
    print("internet: NOT reachable. Settings > Internet > On, then re-run this cell.")"""

DATASET_KAGGLE_MD = """\
## The dataset

The dataset is read in place from `/kaggle/input`, so nothing is downloaded and no key is needed. If the cell below cannot find it, use **+ Add Input** in the right panel and add `faceforensics-c23-extracted-faces-100k`."""

DATASET_KAGGLE_CODE = """\
import sys

sys.path.insert(0, "/kaggle/working")   # the package written by the module cells
from dfv2.manifest import find_data_dir

DATA_DIR = str(find_data_dir("/kaggle/input"))
print("dataset:", DATA_DIR)
print(sorted(os.listdir(DATA_DIR)))"""

SETTINGS_COLAB_MD = """\
## Setup and settings

Constants you can change. Everything is written to Google Drive, so a disconnect loses nothing."""

SETTINGS_COLAB_CODE = """\
import os
import subprocess
import sys

import torch

""" + RUN_HELPER + """

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    os.chdir("/content")
os.makedirs("dfv2", exist_ok=True)          # the code cells below write the package here

DATA_DIR = "/content/dataset/dataset_processed_split"   # where the dataset gets unzipped
OUT_DIR = "/content/drive/MyDrive/deepfake_v2_runs"     # on Drive, so a disconnect loses nothing
SEEDS = [0]            # start with [0] (4 runs); use [0, 1, 2] for the full study
EPOCHS = 5
SPLIT = "clean"        # identity-disjoint; "original" is Kaggle's assignment (use a different OUT_DIR)
EXTRA_FLAGS = ""       # e.g. "--no-amp" if mixed precision errors in the quick test

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU. On CPU this would take days.")"""

DRIVE_MD = """\
## Google Drive

Results are written to Drive so that nothing is lost if the session disconnects. Outside Colab this step is skipped and results go to `./runs`."""

DRIVE_CODE = """\
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
else:
    OUT_DIR = "runs\""""

KAGGLE_DOWNLOAD_MD = """\
## The dataset

The data comes from Kaggle (`gradientvoyager/faceforensics-c23-extracted-faces-100k`, about 2.5 GB, CC-BY-NC-SA-4.0).

**Credentials.** In Colab open the key icon in the left sidebar (Secrets) and add two secrets, `KAGGLE_USERNAME` and `KAGGLE_KEY`, with "Notebook access" switched on. Never paste a key into a cell: a notebook with a key in it exposes the key as soon as the notebook is shared or uploaded. If a key was ever pasted into an earlier notebook, revoke it on Kaggle and create a new one."""

KAGGLE_DOWNLOAD_CODE = """\
manifest = os.path.join(DATA_DIR, "dataset_manifest.csv")
if os.path.exists(DATA_DIR):
    print("dataset already present")
else:
    if IN_COLAB:
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    assert os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"), \\
        "Set KAGGLE_USERNAME and KAGGLE_KEY (see the note above)."
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    subprocess.check_call(["kaggle", "datasets", "download", "-d",
                           "gradientvoyager/faceforensics-c23-extracted-faces-100k", "-p", "/content"])
    subprocess.check_call(["unzip", "-q", "-o", "/content/faceforensics-c23-extracted-faces-100k.zip",
                           "-d", "/content/dataset"])
assert os.path.isdir(DATA_DIR), f"dataset not found at {DATA_DIR}; compare DATA_DIR with `ls /content/dataset`\""""

MODULES_MD = """\
## The code

Each cell below writes one file of the `dfv2` package into the working folder. Running them defines everything the later cells use, and you can read or change any of them. These are the files the project's automated tests run against."""

CHECK_SPLIT_MD = """\
## Check the split

Prints how many identities appear in more than one split, for the original Kaggle split and for the clean one. The clean split should show 0, 0, 0."""

CHECK_SPLIT_CODE = 'run("-m", "dfv2.make_split", "--data-dir", DATA_DIR)'

QUICK_MD = """\
## Quick test on the GPU (a few minutes)

Trains one tiny hybrid run on 8 videos per split and label. The accuracy means nothing; this only checks that everything runs on the GPU, including mixed precision. If it fails with a CUDA or mixed-precision error, set `EXTRA_FLAGS = "--no-amp"` in the settings cell, re-run that cell, then this one."""

QUICK_CODE = ('run("-m", "dfv2.run_ablation", "--data-dir", DATA_DIR, "--out-dir", "smoke", "--variants", "hybrid", '
              '"--seeds", "0", "--epochs", "1", "--max-videos", "8", "--split", SPLIT, *EXTRA_FLAGS.split())')

EXPERIMENT_CODE = """\
seeds = [str(s) for s in SEEDS]
run("-m", "dfv2.run_ablation", "--data-dir", DATA_DIR, "--out-dir", OUT_DIR, "--seeds", *seeds,
    "--epochs", str(EPOCHS), "--split", SPLIT, *EXTRA_FLAGS.split())"""

REPORT_MD = """\
## The report

Reads the saved predictions and writes `report.md`, `results.json` and three figures into `<OUT_DIR>/report/`. No GPU needed; about a minute."""

REPORT_CODE = """\
run("-m", "dfv2.report", "--runs-dir", OUT_DIR)

from IPython.display import Image, Markdown, display
report_dir = os.path.join(OUT_DIR, "report")
display(Markdown(open(os.path.join(report_dir, "report.md")).read()))
for name in ["roc_frame.png", "per_manipulation_auc.png", "learning_curves.png"]:
    display(Image(filename=os.path.join(report_dir, name)))"""

HOW_TO_READ = """\
## How to read it

1. Start with **What the data supports**. Each line is a difference with a 95% interval; "not resolved" means the test set cannot tell the effect from zero. That is an honest result, not a failure.
2. **Component effects** separate the gate, the FFT branch and their interaction, using the same resampled videos on both sides.
3. **By manipulation type** shows whether a gain is spread across methods or comes from one.
4. Read the **caveats** before quoting any number.

When the runs with seeds 0, 1 and 2 are done, the results are ready to replace the old claims (93.34%, the Real-precision explanation) in the notebooks and the README.

Not covered here: other datasets, other compression levels, a redesigned frequency branch, and repeating the study over several split seeds (`--split-seed`)."""


def experiment_md(target: str) -> str:
    if target == "colab":
        tail = ("Keep the tab open, or the session may disconnect. If it does, run the cells again: "
                "finished runs are skipped.")
    else:
        tail = ("A Kaggle notebook session lasts at most 12 hours. To run everything unattended, use "
                "**Save Version > Save & Run All (Commit)** instead of running cells by hand. That version keeps "
                "everything under `/kaggle/working`, including `runs/`.")
    return ("## The experiment\n\n"
            "Four variants x the seeds in `SEEDS`. Start with `SEEDS = [0]` (four runs), check the report, then set "
            "`[0, 1, 2]` and run this cell again: finished runs are skipped.\n\n"
            "Each run trains for `EPOCHS` epochs on a fresh balanced real/fake subset every epoch, keeps the epoch "
            "with the best validation AUC, and reads the test split once. Time per run has not been measured. A guess "
            "is 10 to 20 minutes on a T4, so the full 12 runs could take a few hours. " + tail)


def lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(text)}


def module_cells() -> list[dict]:
    cells = []
    for name, blurb in MODULES:
        cells.append(md(f"### `dfv2/{name}`\n\n{blurb}"))
        body = (ROOT / "dfv2" / name).read_text().rstrip("\n")
        cells.append(code(f"%%writefile dfv2/{name}\n{body}"))
    return cells


def run_cells(target: str) -> list[dict]:
    return [md(CHECK_SPLIT_MD), code(CHECK_SPLIT_CODE), md(QUICK_MD), code(QUICK_CODE),
            md(experiment_md(target)), code(EXPERIMENT_CODE), md(REPORT_MD), code(REPORT_CODE), md(HOW_TO_READ)]


def kaggle_cells() -> list[dict]:
    return ([md(TITLE_KAGGLE), md(IDEA), md(SETTINGS_KAGGLE_MD), code(SETTINGS_KAGGLE_CODE), md(MODULES_MD)]
            + module_cells()
            + [md(DATASET_KAGGLE_MD), code(DATASET_KAGGLE_CODE)]
            + run_cells("kaggle"))


def colab_cells() -> list[dict]:
    return ([md(TITLE_COLAB), md(IDEA), md(SETTINGS_COLAB_MD), code(SETTINGS_COLAB_CODE),
             md(DRIVE_MD), code(DRIVE_CODE), md(KAGGLE_DOWNLOAD_MD), code(KAGGLE_DOWNLOAD_CODE), md(MODULES_MD)]
            + module_cells()
            + run_cells("colab"))


def notebook(cells: list[dict], colab_name: str | None = None) -> dict:
    metadata = {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if colab_name:
        metadata["colab"] = {"name": colab_name, "provenance": [], "gpuType": "T4"}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 4}


TARGETS = {
    "kaggle": ("Deepfake_Detection_v2_kaggle.ipynb", lambda: notebook(kaggle_cells())),
    "colab": ("Deepfake_Detection_v2.ipynb", lambda: notebook(colab_cells(), "Deepfake_Detection_v2.ipynb")),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=[*TARGETS, "all"], default="all")
    p.add_argument("--out-dir", default=str(ROOT.parent), help="where to write the notebooks (default: project folder)")
    args = p.parse_args()
    names = list(TARGETS) if args.target == "all" else [args.target]
    for name in names:
        filename, build = TARGETS[name]
        out = Path(args.out_dir) / filename
        out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
