"""
Filters an existing consolidated YOLO dataset (all 6 classes: Home, First,
Second, Third, Player, Ignore) down to a single-class Player-only dataset.

Keeps only label lines with class_id == 4 (Player), remaps them to class_id 0,
and copies images unchanged (images don't need to change, only labels).
"""
import argparse
import os
import shutil


def filter_split(images_in_dir, labels_in_dir, images_out_dir, labels_out_dir,
                  player_class_id):
    '''
    Filters one split (train or val): copies every image unchanged, and
    rewrites its label file to keep only Player boxes remapped to class 0.

    Returns:
        n_frames : int, total frames processed
        n_frames_with_players : int, frames that had >=1 player box
        n_player_boxes : int, total player boxes kept across all frames
    '''
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    n_frames = 0
    n_frames_with_players = 0
    n_player_boxes = 0

    for fname in sorted(os.listdir(images_in_dir)):
        if not fname.lower().endswith('.png'):
            continue

        stem = os.path.splitext(fname)[0]
        src_image_path = os.path.join(images_in_dir, fname)
        src_label_path = os.path.join(labels_in_dir, stem + '.txt')

        dst_image_path = os.path.join(images_out_dir, fname)
        dst_label_path = os.path.join(labels_out_dir, stem + '.txt')

        # Copy image unchanged
        shutil.copyfile(src_image_path, dst_image_path)

        # Filter and remap label lines
        kept_lines = []
        if os.path.exists(src_label_path):
            with open(src_label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    if class_id == player_class_id:
                        # Remap to class 0 since Player is our only class now
                        remapped_line = ' '.join(['0'] + parts[1:])
                        kept_lines.append(remapped_line)

        with open(dst_label_path, 'w') as f:
            if kept_lines:
                f.write('\n'.join(kept_lines) + '\n')
            

        n_frames += 1
        if kept_lines:
            n_frames_with_players += 1
            n_player_boxes += len(kept_lines)

    return n_frames, n_frames_with_players, n_player_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
        help='Path to existing 6-class consolidated dataset (with images/, labels/, data.yaml)')
    parser.add_argument('--output_dir', type=str, required=True,
        help='Path to write the new Player-only dataset to')
    parser.add_argument('--player_class_id', type=int, default=4,
        help='Class id for Player in the original 6-class data.yaml (default 4)')
    args = parser.parse_args()

    for split in ['train', 'val']:
        images_in_dir = os.path.join(args.input_dir, 'images', split)
        labels_in_dir = os.path.join(args.input_dir, 'labels', split)
        images_out_dir = os.path.join(args.output_dir, 'images', split)
        labels_out_dir = os.path.join(args.output_dir, 'labels', split)

        print('Filtering {} split...'.format(split))
        n_frames, n_with_players, n_boxes = filter_split(
            images_in_dir, labels_in_dir, images_out_dir, labels_out_dir,
            args.player_class_id)

        print('  {} frames processed'.format(n_frames))
        print('  {} frames contain >=1 player box'.format(n_with_players))
        print('  {} frames have ZERO player boxes (all-negative frames)'.format(
            n_frames - n_with_players))
        print('  {} total player boxes kept'.format(n_boxes))
        print('  avg player boxes per positive frame: {:.2f}'.format(
            n_boxes / max(1, n_with_players)))

    # Copy train.txt / val.txt as-is (paths/filenames don't change)
    for txt_name in ['train.txt', 'val.txt']:
        src = os.path.join(args.input_dir, txt_name)
        dst = os.path.join(args.output_dir, txt_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    # Write new single-class data.yaml
    data_yaml_path = os.path.join(args.output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write('path: .\n')
        f.write('train: train.txt\n')
        f.write('val: val.txt\n')
        f.write('names:\n')
        f.write('  0: Player\n')

    print('\nWrote {}'.format(data_yaml_path))
    print('Done. Player-only dataset at: {}'.format(args.output_dir))


if __name__ == '__main__':
    main()