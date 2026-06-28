"""Put eval/ on sys.path so `import evallib` works under pytest from the repo root."""
import pathlib
import sys

EVAL_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
