from pathlib import Path

# Get repo root (two levels up from pipeline script)
repo_root = Path(__file__).resolve().parents[3]  # adjust levels if needed

CONFIG_FILE_PATH = repo_root / "config" / "config.yaml"
PARAMS_FILE_PATH = repo_root / "params.yaml"
