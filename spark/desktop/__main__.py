"""``python -m spark.desktop``: the desktop window (also the installer's entry point)."""

import sys

from spark.desktop.window import main

if __name__ == "__main__":
    sys.exit(main())
