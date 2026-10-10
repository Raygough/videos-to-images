"""Test path setup: make repo-root and detector/ importable, add helper factories."""

import os
import sys

import cv2
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "detector"))


@pytest.fixture
def tiny_video(tmp_path):
    """Write a small, deterministic MP4 via OpenCV and return its path.

    The clip is 12 frames of smooth grayscale gradients so perceptual hashes
    actually vary frame to frame.
    """
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps=10,
        frameSize=(64, 48),
    )
    try:
        for n in range(12):
            frame = np.full((48, 64, 3), n * 18, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    assert os.path.getsize(path) > 0
    return str(path)


@pytest.fixture
def sample_png(tmp_path):
    """Write a deterministic PNG and return its path (for hashing tests)."""
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.new("L", (8, 8), 100).save(path)
    return str(path)