import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# gdpo/ (rewards) and eval/ (eval_miou) are plain modules on these paths.
for sub in ("gdpo", "eval"):
    path = os.path.join(REPO_ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)
