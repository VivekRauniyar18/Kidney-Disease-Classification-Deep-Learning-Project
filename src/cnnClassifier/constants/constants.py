from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # goes to root of project

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"
