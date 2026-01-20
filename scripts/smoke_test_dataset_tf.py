# main/scripts/smoke_test_dataset_tf.py
from src.common.paths import DATA_LA, ARTIFACTS
from src.dataio.protocols import build_manifest_la
from src.features.normalize import NormStats
from src.dataio.dataset_tf import make_dataset, DatasetConfig

def main():
    stats_path = ARTIFACTS / "feature_stats" / "stats_train3000_v1.npz"
    stats = NormStats.load(stats_path)

    train_samples = build_manifest_la(DATA_LA, split="train")[:200]
    ds_cfg = DatasetConfig(batch_size=8, shuffle=True)

    ds = make_dataset(train_samples, ds_cfg, stats)
    batch = next(iter(ds))
    x, y = batch
    print("mel_in:", x["mel_in"].shape)  # [B, 80, T, 1]
    print("seq_in:", x["seq_in"].shape)  # [B, T, 60]
    print("y:", y.shape)

if __name__ == "__main__":
    main()
