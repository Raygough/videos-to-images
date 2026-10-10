"""Shared hashing, matching and YAML helpers for the video-frame pipeline.

Several scripts in :mod:`detector` independently re-implemented perceptual
hashing and nearest-hash lookup. This module is the single home for those
helpers so the behaviour stays consistent (and is unit-testable without
touching a dataset on disk).

The image-reading functions require Pillow; the pure helpers
(:func:`hash_pixels`, :func:`hamming_distance_hex`, :func:`extract_frame_number`,
:func:`parse_yaml_names`) require only the standard library, so they can be
tested in isolation.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

__all__ = [
    "hash_pixels",
    "average_hash",
    "hamming_distance_hex",
    "extract_frame_number",
    "find_nearest_hash",
    "parse_yaml_names",
    "read_yaml_class_names",
]


def hash_pixels(pixels: Iterable[int], hash_size: int) -> str:
    """Average-hash a flat sequence of grayscale pixel values.

    Resize to ``hash_size`` is the caller's job; this helper takes the already
    downsampled pixel stream. Thresholds against the mean and packs the
    comparison results into a 256-bit integer expressed as a hex string.

    Args:
        pixels: Flat iterable of ``hash_size * hash_size`` pixel values.
        hash_size: Width/height of the (already resized) image.

    Returns:
        The perceptual hash as a hex string such as ``0xabc...``.
    """
    px = list(pixels)
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p > avg:
            bits |= 1 << i
    return hex(bits)


def average_hash(image_path: str, hash_size: int = 16) -> str:
    """Compute the average perceptual hash of an image file.

    Resize -> grayscale -> compare each pixel to the mean -> pack bits.
    Robust to minor recompression differences (e.g. PNG re-encoding),
    unlike a raw byte/MD5 hash.
    """
    from PIL import Image  # local import keeps unborn PIL off the pure path

    resample = getattr(Image.Resampling, "LANCZOS", None) or Image.LANCZOS
    img = Image.open(image_path).convert("L").resize(
        (hash_size, hash_size), resample
    )
    # Prefer the modern flattened-data API; fall back to getdata() on Pillow < 13.
    if hasattr(img, "get_flattened_data"):
        pixels = img.get_flattened_data()
    else:
        pixels = img.getdata()
    return hash_pixels(pixels, hash_size)


def hamming_distance_hex(h1: str, h2: str) -> int:
    """Number of differing bits between two hex-encoded hashes."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def extract_frame_number(filename: str) -> Optional[int]:
    """Return the first integer found in ``filename``, or ``None``.

    Used to order raw frames by their frame number so a windowed MSE search
    can reason about neighbours.
    """
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else None


def find_nearest_hash(
    frame_hash: str,
    index: dict,
    max_hamming_distance: int = 10,
) -> tuple[Optional[list], Optional[int]]:
    """Locate the closest entry for ``frame_hash`` in a hash index.

    Tries an exact lookup first (fast path); on a miss falls back to a
    brute-force nearest-neighbour scan by hamming distance. This matches the
    semantics previously duplicated across ``match_job_to_sequence.py`` and
    ``build_full_frame_mapping_v3.py`` (the latter defaulting to a larger
    distance bound for its rougher candidate stage).

    Args:
        frame_hash: Hex-encoded query hash.
        index: ``{hash_hex: [[sequence_id, frame_filename], ...], ...}``.
        max_hamming_distance: Reject matches whose distance exceeds this.

    Returns:
        A ``(entry, distance)`` tuple where ``entry`` is the first
        ``[sequence_id, frame_filename]`` pair from the match (or ``None``)
        and ``distance`` is the hamming distance to it (or ``None``).
    """
    if frame_hash in index:
        # Exact match -- take the first (sequence_id, frame_filename) entry.
        return index[frame_hash][0], 0

    best_entry: Optional[list] = None
    best_distance: Optional[int] = None
    for candidate_hash, entries in index.items():
        d = hamming_distance_hex(frame_hash, candidate_hash)
        if best_distance is None or d < best_distance:
            best_distance = d
            best_entry = entries[0]

    if best_entry is not None and best_distance is not None and best_distance <= max_hamming_distance:
        return best_entry, best_distance
    return None, None


def parse_yaml_names(lines: Iterable[str]) -> dict[int, str]:
    """Parse a ``names:`` block out of small YOLO/data YAML without PyYAML.

    Reads lines like ``  0: Home`` / ``  2: First`` into
    ``{0: "Home", 2: "First"}``. Stops at the first non-mapping line after the
    block begins, so it works whether the block is the last thing in the file
    or is followed by further keys.

    Args:
        lines: Iterable of the raw file lines.

    Returns:
        Mapping of class id -> class name.
    """
    names: dict[int, str] = {}
    in_names_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names_block = True
            continue
        if in_names_block:
            m = re.match(r"(\d+):\s*(.+)", stripped)
            if m:
                names[int(m.group(1))] = m.group(2)
            else:
                break  # end of the names block
    return names


def read_yaml_class_names(data_yaml_path: str) -> dict[int, str]:
    """Read class names from a ``data.yaml`` file by path."""
    with open(data_yaml_path, "r") as f:
        return parse_yaml_names(f)