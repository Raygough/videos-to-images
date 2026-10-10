"""
Sanity-checks the fuzzy-matched frame mapping in two ways:
1. Prints the distribution of hamming distances (are they all small/tight,
   or spread out -- spread out is more concerning).
2. Saves side-by-side images (queried detection frame next to its matched
   raw frame) for a sample of matches at different distance levels, so you
   can visually confirm whether fuzzy matches are actually correct.

Usage:
    python verify_frame_mapping.py \
        --dataset_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/dataset_player_only \
        --frames_dir /nfs/roberts/pi/pi_aw989/rjg73/frames \
        --mapping_path /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/frame_to_sequence_map.json \
        --output_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/mapping_spot_check \
        --n_samples_per_bucket 4
"""
import os
import json
import random
import argparse
from collections import Counter
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--mapping_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_samples_per_bucket', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with open(args.mapping_path, 'r') as f:
        mapping = json.load(f)

    # --- Part 1: distance distribution ---
    distances = [v['hamming_distance'] for v in mapping.values()
                 if v['hamming_distance'] is not None]

    distance_counts = Counter(distances)
    print('Hamming distance distribution:')
    for d in sorted(distance_counts.keys()):
        print('  distance {}: {} frames'.format(d, distance_counts[d]))

    print('\nMin: {}, Max: {}, Mean: {:.2f}'.format(
        min(distances), max(distances), sum(distances) / len(distances)))

    # --- Part 2: visual spot check across distance buckets ---
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Bucket into low (1-2), mid (3-5), high (6+) distance groups
    buckets = {'low': [], 'mid': [], 'high': []}
    for fname, v in mapping.items():
        d = v['hamming_distance']
        if d is None:
            continue
        if d == 0:
            continue  # skip exact matches, nothing to verify there
        elif d <= 2:
            buckets['low'].append(fname)
        elif d <= 5:
            buckets['mid'].append(fname)
        else:
            buckets['high'].append(fname)

    for bucket_name, fnames in buckets.items():
        print('\nBucket "{}": {} frames available'.format(bucket_name, len(fnames)))
        n_sample = min(args.n_samples_per_bucket, len(fnames))
        if n_sample == 0:
            continue

        sampled = random.sample(fnames, n_sample)
        for fname in sampled:
            entry = mapping[fname]
            split = entry['split']
            seq_id = entry['sequence_id']
            source_fname = entry['source_frame_filename']
            distance = entry['hamming_distance']

            query_path = os.path.join(args.dataset_dir, 'images', split, fname)
            match_path = os.path.join(args.frames_dir, seq_id, source_fname)

            if not os.path.exists(match_path):
                print('  WARNING: matched source frame missing on disk: {}'.format(match_path))
                continue

            query_img = Image.open(query_path).convert('RGB')
            match_img = Image.open(match_path).convert('RGB')

            # Resize both to same height for a clean side-by-side comparison
            target_h = 480
            def resize_to_height(img, h):
                w = int(img.width * (h / img.height))
                return img.resize((w, h))

            query_img = resize_to_height(query_img, target_h)
            match_img = resize_to_height(match_img, target_h)

            combined = Image.new('RGB', (query_img.width + match_img.width, target_h))
            combined.paste(query_img, (0, 0))
            combined.paste(match_img, (query_img.width, 0))

            out_name = '{}_d{}_{}_vs_{}.png'.format(bucket_name, distance, fname, source_fname)
            out_path = os.path.join(args.output_dir, out_name)
            combined.save(out_path)
            print('  Saved {} (distance={}, sequence={})'.format(out_path, distance, seq_id))

    print('\nDone. Open the images in {} -- LEFT = labeled detection frame, '
          'RIGHT = matched raw frame. They should show the same scene.'.format(
              args.output_dir))


if __name__ == '__main__':
    main()