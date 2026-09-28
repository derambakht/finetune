"""Launch Unsloth FineTune Studio without installing the package.

    python app.py                 # http://localhost:7860
    python app.py --share         # public link (Colab / remote servers)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ftstudio.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["ui", *sys.argv[1:]]))
