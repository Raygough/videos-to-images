"""Extract image frames from videos (SLURM-friendly entry point).

This was a separate, hardcoded copy of :file:`main.py` that also contained a
stale ``videos = videos[342:]`` slice (a leftover from a specific data run)
which silently ignored the first 342 videos. It now delegates to the shared
engine in :mod:`extractor` and processes every video.

See ``extractor.py --help`` for the accepted flags.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import run_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cli())