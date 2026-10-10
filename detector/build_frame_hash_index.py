"""
Builds an index mapping a lightweight perceptual hash of each raw extracted
frame to its (sequence_id, frame_filename). Used later to identify which
video sequence a CVAT-exported frame originally came from, since job zips
carry no metadata linking them back to a sequence id.

Uses a simple average-hash (aHash): resize to 16x16 grayscale, threshold
against the mean, pack into a 256-bit integer. Robust to minor recompression
differences (e.g. if CVAT re-encoded the PNG), unlike a raw byte/MD5 hash.

Usage:
    python build_frame_hash_index.py \
        --frames_dir /nfs/roberts/pi/pi_aw989/rjg73/frames \
        --output_path /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/frame_hash_index.json
"""
import os
import json
import argparse
from PIL import Image


def average_hash(image_path, hash_size=16):
    '''
    Computes a simple average hash of an image as a hex string.
    Resize -> grayscale -> compare each pixel to the mean -> pack bits.
    '''
    img = Image.open(image_path).convert('L').resize(
        (hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)

    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= (1 << i)

    return hex(bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, required=True,
        help='Directory containing one subfolder per sequence id, each with frame images')
    parser.add_argument('--output_path', type=str, required=True,
        help='Where to write the resulting JSON hash index')
    parser.add_argument('--hash_size', type=int, default=16)
    args = parser.parse_args()

    sequence_ids = sorted([
        d for d in os.listdir(args.frames_dir)
        if os.path.isdir(os.path.join(args.frames_dir, d))
    ])

    print('Found {} sequences to index'.format(len(sequence_ids)))

    # index: hash_hex -> list of [sequence_id, frame_filename]
    # (list, not single value, in case of hash collisions across sequences)
    index = {}

    for seq_idx, sequence_id in enumerate(sequence_ids):
        seq_dir = os.path.join(args.frames_dir, sequence_id)
        frame_files = sorted([
            f for f in os.listdir(seq_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        for fname in frame_files:
            frame_path = os.path.join(seq_dir, fname)
            try:
                h = average_hash(frame_path, args.hash_size)
            except Exception as e:
                print('  Skipping unreadable frame {}: {}'.format(frame_path, e))
                continue

            index.setdefault(h, []).append([sequence_id, fname])

        if (seq_idx + 1) % 20 == 0 or seq_idx == len(sequence_ids) - 1:
            print('Indexed {}/{} sequences ({} unique hashes so far)'.format(
                seq_idx + 1, len(sequence_ids), len(index)))

    with open(args.output_path, 'w') as f:
        json.dump(index, f)

    print('\nWrote hash index with {} unique hashes to {}'.format(
        len(index), args.output_path))


if __name__ == '__main__':
    main()