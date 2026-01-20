# main/scripts/make_balanced_subsets.py
from __future__ import annotations

from src.common.paths import DATA_LA, ARTIFACTS
from src.dataio.protocols import build_manifest_la
from src.dataio.manifests import stratified_sample, save_manifest_csv

def main():
    SEED = 7
    train_n = 3000
    dev_n = 800
    test_n = 800

    train_all = build_manifest_la(DATA_LA, split="train")
    dev_all = build_manifest_la(DATA_LA, split="dev")
    eval_all = build_manifest_la(DATA_LA, split="eval")

    train = stratified_sample(train_all, train_n, seed=SEED)
    dev = stratified_sample(dev_all, dev_n, seed=SEED + 1)
    test = stratified_sample(eval_all, test_n, seed=SEED + 2)

    out_dir = ARTIFACTS / "manifests"
    save_manifest_csv(train, out_dir / f"train_{train_n}_balanced.csv")
    save_manifest_csv(dev, out_dir / f"dev_{dev_n}_balanced.csv")
    save_manifest_csv(test, out_dir / f"test_{test_n}_balanced.csv")

    print("Saved manifests to:", out_dir)

if __name__ == "__main__":
    main()
