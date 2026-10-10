"""
Extract video data into a single train/val dataset 
with job-prefixed filenames (to avoid collisions across jobs),
 and writes data.yaml + train.txt + val.txt.

Split is done at the JOB level (sequence/video level), not frame level,
since frames within one job are highly correlated.


python prepare_detection_dataset.py \
    --zip_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/raw_zips \
    --output_dir /nfs/roberts/pi/pi_aw989/rjg73/data/tigers_detection/dataset \
    --val_fraction 0.2 \
    --seed 42
"""
import os
import re
import shutil
import zipfile
import argparse
import random


def extract_job_id(zip_path):
    """Extract job id from filename like dataset_job_3844683.zip"""
    match = re.search(r'job_(\d+)', os.path.basename(zip_path))
    if match is None:
        raise ValueError('Could not parse job id from {}'.format(zip_path))
    return match.group(1)


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)


def find_data_yaml_class_names(extract_to):
    '''
    Reads names: block out of data.yaml without requiring pyyaml,
    since these files are small and simply formatted.
    '''
    yaml_path = os.path.join(extract_to, 'data.yaml')
    names = {}
    with open(yaml_path, 'r') as f:
        lines = f.readlines()

    in_names_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('names:'):
            in_names_block = True
            continue
        if in_names_block:
            # Lines like "  0: Home"
            m = re.match(r'(\d+):\s*(.+)', stripped)
            if m:
                names[int(m.group(1))] = m.group(2)
            else:
                # Reached end of names block
                break
    return names


def copy_job_frames(job_id, extract_to, images_out_dir, labels_out_dir):
    '''
    Copies all images/train/*.png and labels/train/*.txt from an extracted
    job into the consolidated output dirs, prefixing filenames with the
    job id so frames from different jobs never collide.

    Returns number of frames copied.
    '''
    src_images_dir = os.path.join(extract_to, 'images', 'train')
    src_labels_dir = os.path.join(extract_to, 'labels', 'train')

    if not os.path.isdir(src_images_dir):
        raise FileNotFoundError('Expected {} to exist'.format(src_images_dir))

    n_copied = 0
    for fname in sorted(os.listdir(src_images_dir)):
        if not fname.lower().endswith('.png'):
            continue

        frame_stem = os.path.splitext(fname)[0]  # e.g. frame_000000
        new_stem = 'job{}_{}'.format(job_id, frame_stem)

        src_image_path = os.path.join(src_images_dir, fname)
        src_label_path = os.path.join(src_labels_dir, frame_stem + '.txt')

        dst_image_path = os.path.join(images_out_dir, new_stem + '.png')
        dst_label_path = os.path.join(labels_out_dir, new_stem + '.txt')

        shutil.copyfile(src_image_path, dst_image_path)

        if os.path.exists(src_label_path):
            shutil.copyfile(src_label_path, dst_label_path)
        else:
            # No label file for this frame -- write an empty one so YOLO
            # treats it as a valid negative (no objects) rather than erroring
            open(dst_label_path, 'w').close()

        n_copied += 1

    return n_copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', type=str, required=True,
        help='Directory containing dataset_job_*.zip files')
    parser.add_argument('--output_dir', type=str, required=True,
        help='Directory to write consolidated dataset to')
    parser.add_argument('--val_fraction', type=float, default=0.2,
        help='Fraction of jobs (sequences) to hold out for validation')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for job-level train/val split')
    args = parser.parse_args()

    raw_extract_dir = os.path.join(args.output_dir, '_raw_extracted')
    images_train_dir = os.path.join(args.output_dir, 'images', 'train')
    images_val_dir = os.path.join(args.output_dir, 'images', 'val')
    labels_train_dir = os.path.join(args.output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(args.output_dir, 'labels', 'val')

    for d in [raw_extract_dir, images_train_dir, images_val_dir,
              labels_train_dir, labels_val_dir]:
        os.makedirs(d, exist_ok=True)

    zip_paths = sorted([
        os.path.join(args.zip_dir, f)
        for f in os.listdir(args.zip_dir)
        if f.endswith('.zip')
    ])

    if len(zip_paths) == 0:
        raise ValueError('No zip files found in {}'.format(args.zip_dir))

    print('Found {} job zips'.format(len(zip_paths)))

    # Extract every job zip into its own subfolder first
    job_ids = []
    class_names = None
    for zip_path in zip_paths:
        job_id = extract_job_id(zip_path)
        job_ids.append(job_id)

        job_extract_dir = os.path.join(raw_extract_dir, 'job_{}'.format(job_id))
        if not os.path.isdir(job_extract_dir):
            print('Extracting job {}...'.format(job_id))
            extract_zip(zip_path, job_extract_dir)
        else:
            print('Job {} already extracted, skipping'.format(job_id))

        job_class_names = find_data_yaml_class_names(job_extract_dir)
        if class_names is None:
            class_names = job_class_names
        elif class_names != job_class_names:
            raise ValueError(
                'Class name mismatch between jobs. Job {} has {} but '
                'expected {}'.format(job_id, job_class_names, class_names))

    print('Class names (consistent across all jobs): {}'.format(class_names))

    # Job-level train/val split
    random.seed(args.seed)
    shuffled_job_ids = job_ids[:]
    random.shuffle(shuffled_job_ids)

    n_val = max(1, round(len(shuffled_job_ids) * args.val_fraction))
    val_job_ids = set(shuffled_job_ids[:n_val])
    train_job_ids = set(shuffled_job_ids[n_val:])

    print('Train jobs ({}): {}'.format(len(train_job_ids), sorted(train_job_ids)))
    print('Val jobs ({}): {}'.format(len(val_job_ids), sorted(val_job_ids)))

    # Copy frames into consolidated train/ val/ dirs
    n_train_frames = 0
    n_val_frames = 0

    for job_id in job_ids:
        job_extract_dir = os.path.join(raw_extract_dir, 'job_{}'.format(job_id))

        if job_id in val_job_ids:
            n = copy_job_frames(job_id, job_extract_dir, images_val_dir, labels_val_dir)
            n_val_frames += n
        else:
            n = copy_job_frames(job_id, job_extract_dir, images_train_dir, labels_train_dir)
            n_train_frames += n

        print('Job {}: {} frames -> {}'.format(
            job_id, n, 'val' if job_id in val_job_ids else 'train'))

    print('Total train frames: {}'.format(n_train_frames))
    print('Total val frames: {}'.format(n_val_frames))

    # Write train.txt / val.txt
    train_txt_path = os.path.join(args.output_dir, 'train.txt')
    val_txt_path = os.path.join(args.output_dir, 'val.txt')

    with open(train_txt_path, 'w') as f:
        for fname in sorted(os.listdir(images_train_dir)):
            f.write('images/train/{}\n'.format(fname))

    with open(val_txt_path, 'w') as f:
        for fname in sorted(os.listdir(images_val_dir)):
            f.write('images/val/{}\n'.format(fname))

    # Write consolidated data.yaml
    data_yaml_path = os.path.join(args.output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write('path: .\n')
        f.write('train: train.txt\n')
        f.write('val: val.txt\n')
        f.write('names:\n')
        for class_id in sorted(class_names.keys()):
            f.write('  {}: {}\n'.format(class_id, class_names[class_id]))

    print('\nWrote {}'.format(data_yaml_path))
    print('Wrote {}'.format(train_txt_path))
    print('Wrote {}'.format(val_txt_path))
    print('\nDone. Consolidated dataset at: {}'.format(args.output_dir))


if __name__ == '__main__':
    main()