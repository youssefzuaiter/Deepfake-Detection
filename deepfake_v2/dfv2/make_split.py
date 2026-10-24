"""Show how much identity leakage the original split has and what the clean split looks like.

    python -m dfv2.make_split --data-dir /content/dataset/dataset_processed_split

Nothing is written: the clean split is recomputed from the manifest and the seed whenever it
is needed, and run_ablation records its audit in every run's history file.
"""
from __future__ import annotations

import argparse

from .manifest import load_manifest
from .splits import DEFAULT_DROP_TYPES, audit_identities, format_audit


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seed", type=int, default=0, help="which identity-disjoint split to draw")
    p.add_argument("--keep-dfd", action="store_true", help="keep DeepFakeDetection in the clean split")
    args = p.parse_args(argv)

    original = load_manifest(args.data_dir)
    clean = load_manifest(args.data_dir, split="clean", split_seed=args.seed,
                          drop_types=() if args.keep_dfd else DEFAULT_DROP_TYPES)
    print(format_audit(audit_identities(original), "Original Kaggle split"))
    print()
    print(format_audit(audit_identities(clean), f"Clean split (seed {args.seed}, "
                       f"DeepFakeDetection {'kept' if args.keep_dfd else 'dropped'})"))


if __name__ == "__main__":
    main()
