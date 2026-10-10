"""
Same matching strategy as build_full_frame_mapping_v2.py (hash for a rough
candidate + pixel-level MSE search to disambiguate near-duplicate frames),
but restructured to avoid redundant disk I/O: since neighboring labeled
frames often search overlapping windows in the same source sequence, this
version caches decoded/downsampled frames per sequence in memory so each
raw frame is only ever loaded and resized once, no matter how many query
frames end up needing it.

Usage:
    python build_full_frame_mapping_v3.py \
        --dataset_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/dataset_player_only \
        --frames_dir /nfs/roberts/pi/pi_aw989/rjg73/frames \
        --hash_index_path /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/frame_hash_index.json \
        --output_path /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/frame_to_sequence_map_v3.json \
        --search_window 200
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from frametools import average_hash, extract_frame_number, find_nearest_hash


def load_small_color_array(image_path, size=96):
    from PIL import Image  # local import keeps the module import-light

    resample = getattr(Image.Resampling, "LANCZOS", None) or Image.LANCZOS
    img = Image.open(image_path).convert('RGB').resize((size, size), resample)
    return np.asarray(img, dtype=np.float32)


def find_candidate_sequence(frame_hash, index, max_hamming_distance=15):
    entry, distance = find_nearest_hash(frame_hash, index, max_hamming_distance)
    if entry is None:
        return None, None, None
    return entry[0], entry[1], distance  # seq_id, source_fname, distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--hash_index_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--search_window', type=int, default=200)
    parser.add_argument('--mse_ok_threshold', type=float, default=50.0)
    parser.add_argument('--array_size', type=int, default=96,
        help='Downsample size for MSE comparison; smaller = faster, less discriminating')
    args = parser.parse_args()

    with open(args.hash_index_path, 'r') as f:
        index = json.load(f)
    print('Loaded hash index with {} unique hashes'.format(len(index)))

    # --- Pass 1: fast hash lookup for every query frame to get a rough anchor ---
    all_queries = []  # list of (fname, split, frame_path)
    for split in ['train', 'val']:
        images_dir = os.path.join(args.dataset_dir, 'images', split)
        for fname in sorted(os.listdir(images_dir)):
            if fname.lower().endswith('.png'):
                all_queries.append((fname, split, os.path.join(images_dir, fname)))

    print('Pass 1/2: hashing {} query frames for rough candidate lookup...'.format(
        len(all_queries)))

    query_info = {}  # fname -> dict(split, seq_id, anchor_fname, hash_distance)
    for i, (fname, split, frame_path) in enumerate(all_queries):
        frame_hash = average_hash(frame_path)
        seq_id, anchor_fname, hash_distance = find_candidate_sequence(frame_hash, index)
        query_info[fname] = {
            'split': split, 'frame_path': frame_path,
            'sequence_id': seq_id, 'anchor_fname': anchor_fname,
            'hash_distance': hash_distance,
        }
        if (i + 1) % 1000 == 0:
            print('  {}/{} hashed'.format(i + 1, len(all_queries)))

    # --- Pass 2: group by sequence, cache frames once, do pixel-level MSE search ---
    # Group queries needing refinement (fuzzy matches only) by sequence id
    by_sequence = defaultdict(list)
    for fname, info in query_info.items():
        if info['sequence_id'] is not None and info['hash_distance'] != 0:
            by_sequence[info['sequence_id']].append(fname)

    print('\nPass 2/2: refining {} fuzzy matches across {} sequences...'.format(
        sum(len(v) for v in by_sequence.values()), len(by_sequence)))

    mapping = {}

    # First, fill in exact hash matches directly (no refinement needed)
    for fname, info in query_info.items():
        if info['sequence_id'] is None:
            mapping[fname] = {
                'split': info['split'], 'sequence_id': None,
                'source_frame_filename': None, 'mse': None,
            }
        elif info['hash_distance'] == 0:
            mapping[fname] = {
                'split': info['split'], 'sequence_id': info['sequence_id'],
                'source_frame_filename': info['anchor_fname'], 'mse': 0.0,
            }

    # Now refine fuzzy matches, one sequence at a time, caching decoded frames
    for seq_idx, (seq_id, fnames_needing_refinement) in enumerate(by_sequence.items()):
        seq_dir = os.path.join(args.frames_dir, seq_id)
        all_frames_in_seq = sorted([
            f for f in os.listdir(seq_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        frame_number_to_fname = {}
        for f in all_frames_in_seq:
            fn = extract_frame_number(f)
            if fn is not None:
                frame_number_to_fname[fn] = f

        # Determine the full range of raw frame numbers this sequence's
        # queries could possibly need, then cache exactly that range once.
        needed_frame_numbers = set()
        for fname in fnames_needing_refinement:
            anchor_idx = extract_frame_number(query_info[fname]['anchor_fname'])
            if anchor_idx is None:
                continue
            lo = anchor_idx - args.search_window
            hi = anchor_idx + args.search_window
            for fn in frame_number_to_fname:
                if lo <= fn <= hi:
                    needed_frame_numbers.add(fn)

        print('Sequence {} ({}/{}): caching {} candidate frames for {} queries'.format(
            seq_id, seq_idx + 1, len(by_sequence),
            len(needed_frame_numbers), len(fnames_needing_refinement)))

        frame_cache = {}
        for fn in needed_frame_numbers:
            candidate_path = os.path.join(seq_dir, frame_number_to_fname[fn])
            try:
                frame_cache[fn] = load_small_color_array(candidate_path, args.array_size)
            except Exception:
                continue

        # Now do the actual MSE search per query, reusing the cache
        for fname in fnames_needing_refinement:
            info = query_info[fname]
            query_arr = load_small_color_array(info['frame_path'], args.array_size)

            best_fn = None
            best_mse = None
            for fn, cached_arr in frame_cache.items():
                mse = float(np.mean((query_arr - cached_arr) ** 2))
                if best_mse is None or mse < best_mse:
                    best_mse = mse
                    best_fn = fn

            if best_fn is None:
                mapping[fname] = {
                    'split': info['split'], 'sequence_id': seq_id,
                    'source_frame_filename': info['anchor_fname'],
                    'mse': None, 'flagged_for_review': True,
                }
            else:
                flagged = best_mse > args.mse_ok_threshold
                mapping[fname] = {
                    'split': info['split'], 'sequence_id': seq_id,
                    'source_frame_filename': frame_number_to_fname[best_fn],
                    'mse': best_mse, 'flagged_for_review': flagged,
                }

    with open(args.output_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    n_flagged = sum(1 for v in mapping.values() if v.get('flagged_for_review'))
    n_no_seq = sum(1 for v in mapping.values() if v['sequence_id'] is None)
    print('\nWrote mapping for {} frames to {}'.format(len(mapping), args.output_path))
    print('{} frames had no candidate sequence'.format(n_no_seq))
    print('{} frames flagged for manual review (high MSE)'.format(n_flagged))


if __name__ == '__main__':
    main()