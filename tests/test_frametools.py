"""Unit tests for detector/frametools.py — pure hashing, matching, YAML parsing."""

import pytest
from frametools import (
    average_hash,
    extract_frame_number,
    find_nearest_hash,
    hamming_distance_hex,
    hash_pixels,
    parse_yaml_names,
    read_yaml_class_names,
)

# --- hash_pixels -----------------------------------------------------------

def test_hash_pixels_flat_gray_is_zero():
    # All pixels equal the mean -> every bit is 0 (strict > comparison).
    assert hash_pixels([5] * 16, 4) == "0x0"


def test_hash_pixels_single_dark_pixel_sets_one_bit():
    pixels = [10] * 16
    pixels[3] = 0  # below mean -> bit 3 cleared
    assert hash_pixels(pixels, 4) != "0x0"


def test_hash_pixels_is_deterministic():
    px = [i % 5 for i in range(16)]
    assert hash_pixels(px, 4) == hash_pixels(list(px), 4)


def test_hash_pixels_ordering_sensitive():
    a = [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9]
    b = [9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0]
    assert hash_pixels(a, 4) != hash_pixels(b, 4)


def test_hash_pixels_hex_prefix():
    assert hash_pixels([1, 2, 3, 4], 2).startswith("0x")


def test_average_hash_roundtrip_deterministic(sample_png):
    assert average_hash(sample_png) == average_hash(sample_png)


def test_average_hash_respects_hash_size(tmp_path):
    from PIL import Image

    # A gradient image has varied pixels, so downsampling to a different size
    # genuinely changes the comparison threshold and thus the hash.
    img = Image.new("L", (32, 32))
    for y in range(32):
        for x in range(32):
            img.putpixel((x, y), (x + y) % 255)
    p = tmp_path / "grad.png"
    img.save(p)
    assert average_hash(str(p), hash_size=8) != average_hash(str(p), hash_size=16)


# --- hamming_distance_hex ---------------------------------------------------

def test_hamming_same_hash_zero():
    h = hex(0b101101)
    assert hamming_distance_hex(h, h) == 0


def test_hamming_single_bit_flip():
    assert hamming_distance_hex(hex(0b0001), hex(0b0011)) == 1


def test_hamming_three_bit_flip():
    assert hamming_distance_hex(hex(0b1111), hex(0b0000)) == 4


def test_hamming_different_lengths_ignores_padding():
    # int() normalises hex, so leading zero nibbles add no distance.
    assert hamming_distance_hex("0x0", "0x0000") == 0


# --- extract_frame_number --------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("frame_00000042.png", 42),
    ("IMG_1234.jpg", 1234),
    ("0001.png", 1),
    ("12_34", 12),  # first integer
    ("no_numbers_here", None),
    ("", None),
])
def test_extract_frame_number(name, expected):
    assert extract_frame_number(name) == expected


# --- find_nearest_hash ------------------------------------------------------

def _index():
    return {
        "0x0": [["seq_a", "f0.png"]],
        "0x1": [["seq_b", "f1.png"]],
    }


def test_find_nearest_exact_match_fast_path():
    entry, dist = find_nearest_hash("0x0", _index())
    assert entry == ["seq_a", "f0.png"]
    assert dist == 0


def test_find_nearest_fuzzy_match_returns_best_and_distance():
    entry, dist = find_nearest_hash("0x3", _index(), max_hamming_distance=2)
    assert entry == ["seq_b", "f1.png"]  # 0x1 is 1 bit from 0x3; 0x0 is 2 bits
    assert dist == 1


def test_find_nearest_exceeds_max_distance_rejected():
    # 0xF is at distance 4 from 0x0 and 3 from 0x1 — all beyond a bound of 1.
    entry, dist = find_nearest_hash("0xF", _index(), max_hamming_distance=1)
    assert entry is None and dist is None


def test_find_nearest_empty_index():
    entry, dist = find_nearest_hash("0x0", {})
    assert entry is None and dist is None


def test_find_nearest_uses_first_entry_on_collision():
    index = {"0x0": [["seq_a", "f0.png"], ["seq_b", "dup.png"]]}
    entry, dist = find_nearest_hash("0x0", index)
    assert entry == ["seq_a", "f0.png"]


# --- parse_yaml_names -------------------------------------------------------

def test_parse_yaml_names_basic():
    lines = [
        "path: .",
        "train: train.txt",
        "names:",
        "  0: Home",
        "  1: First",
        "  4: Player",
        "train: val.txt",
    ]
    assert parse_yaml_names(lines) == {0: "Home", 1: "First", 4: "Player"}


def test_parse_yaml_names_stops_at_first_non_mapping_line():
    lines = ["names:", "  0: Home", "other_key: x", "  2: Ignore"]
    assert parse_yaml_names(lines) == {0: "Home"}


def test_parse_yaml_names_no_names_block():
    assert parse_yaml_names(["path: .", "train: train.txt"]) == {}


def test_parse_yaml_names_crlf_and_space_variants():
    lines = ["names:\r\n", "  0: Home\r\n", "7:  Seven\r\n"]
    assert parse_yaml_names(lines) == {0: "Home", 7: "Seven"}


def test_read_yaml_class_names_from_file(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text("names:\n  2: First\n  3: Second\n")
    assert read_yaml_class_names(str(p)) == {2: "First", 3: "Second"}