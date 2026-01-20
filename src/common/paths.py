# main/src/common/paths.py
from pathlib import Path

def project_root() -> Path:
    # .../main/src/common/paths.py -> parents[2] = .../main
    root = Path(__file__).resolve().parents[2]
    if not (root / "src").exists():
        raise RuntimeError(f"Project root detection failed. Got: {root}")
    return root

ROOT = project_root()
DATA_LA = ROOT / "data" / "LA"
ARTIFACTS = ROOT / "artifacts"
CHECKPOINTS = ROOT / "checkpoints"
CONFIGS = ROOT / "configs"

def ensure_dirs() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
