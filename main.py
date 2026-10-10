"""videos-to-images: extract image frames from videos with OpenCV.

Delegates to the shared engine in :mod:`extractor` (see there for CLI flags:
--step, --skip, --pad, --ext, --only-extensions, --verbose).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli())