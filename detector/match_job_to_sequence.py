"""
For each CVAT job (dataset_job_XXXXXXX.zip, already extracted), samples a
handful of its frames, hashes them the same way as build_frame_hash_index.py,
and looks them up in the index to determine which raw video sequence the
job's frames actually came from.

Outputs a JSON mapping: { job_id: {"sequence_id": ..., "confidence": ...} }

Usage:
    python match_job_to_sequence.py \
        --jobs_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/dataset/_raw_extracted \
        --hash_index_path /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/frame_hash_index.json \
        --output_path /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/job_to_sequence_map.json \
        --n_samples_per_job 8
"""
import os
import re
import json
import random
import argparse
from collections import Counter
from PIL import Image


def average_hash(image_path, hash_size=16):
    img = Image.open(image_path).convert('L').resize(
        (hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)

    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= (1 << i)

    return hex(bits)


def hamming_distance_hex(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count('1')


def find_best_sequence_match(frame_hash, index, max_hamming_distance=10):
    '''
    First tries an exact hash lookup (fast path). If nothing exact, falls
    back to brute-force nearest neighbor by hamming distance across the
    whole index (slow, but only triggered on a miss).
    '''
    if frame_hash in index:
        # Exact match -- take the first (sequence_id, frame_filename) entry
        seq_id, _ = index[frame_hash][0]
        return seq_id, 0

    best_seq_id = None
    best_distance = None
    for candidate_hash, entries in index.items():
        d = hamming_distance_hex(frame_hash, candidate_hash)
        if best_distance is None or d < best_distance:
            best_distance = d
            best_seq_id = entries[0][0]

    if best_distance is not None and best_distance <= max_hamming_distance:
        return best_seq_id, best_distance

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs_dir', type=str, required=True,
        help='Directory containing job_<id>/ subfolders (extracted CVAT zips)')
    parser.add_argument('--hash_index_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--n_samples_per_job', type=int, default=8,
        help='Number of frames to sample per job for voting')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with open(args.hash_index_path, 'r') as f:
        index = json.load(f)

    print('Loaded hash index with {} unique hashes'.format(len(index)))

    random.seed(args.seed)

    job_dirs = sorted([
        d for d in os.listdir(args.jobs_dir)
        if d.startswith('job_') and os.path.isdir(os.path.join(args.jobs_dir, d))
    ])

    results = {}

    for job_dir_name in job_dirs:
        job_id = job_dir_name.replace('job_', '')
        images_dir = os.path.join(args.jobs_dir, job_dir_name, 'images', 'train')

        if not os.path.isdir(images_dir):
            print('Job {}: no images/train dir found, skipping'.format(job_id))
            continue

        frame_files = sorted([
            f for f in os.listdir(images_dir) if f.lower().endswith('.png')
        ])

        n_samples = min(args.n_samples_per_job, len(frame_files))
        sampled_frames = random.sample(frame_files, n_samples)

        votes = Counter()
        for fname in sampled_frames:
            frame_path = os.path.join(images_dir, fname)
            frame_hash = average_hash(frame_path)
            seq_id, distance = find_best_sequence_match(frame_hash, index)
            if seq_id is not None:
                votes[seq_id] += 1

        if votes:
            best_seq_id, n_votes = votes.most_common(1)[0]
            confidence = n_votes / n_samples
            results[job_id] = {
                'sequence_id': best_seq_id,
                'confidence': confidence,
                'votes': dict(votes),
            }
            print('Job {}: matched to sequence {} (confidence {:.2f}, votes {})'.format(
                job_id, best_seq_id, confidence, dict(votes)))
        else:
            results[job_id] = {
                'sequence_id': None,
                'confidence': 0.0,
                'votes': {},
            }
            print('Job {}: NO MATCH FOUND across {} sampled frames'.format(
                job_id, n_samples))

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print('\nWrote job-to-sequence mapping to {}'.format(args.output_path))

    unmatched = [j for j, r in results.items() if r['sequence_id'] is None]
    low_confidence = [j for j, r in results.items()
                       if r['sequence_id'] is not None and r['confidence'] < 0.75]

    if unmatched:
        print('WARNING: {} jobs had no match at all: {}'.format(
            len(unmatched), unmatched))
    if low_confidence:
        print('WARNING: {} jobs matched with <75% vote confidence, worth manually reviewing: {}'.format(
            len(low_confidence), low_confidence))


if __name__ == '__main__':
    main()