"""
Draws YOLO-format bounding boxes onto sample images to visually
confirm labels actually align with the players/objects in the frame,
rather than trusting the raw label numbers.
"""
import argparse
import os
import random

from frametools import read_yaml_class_names
from PIL import Image, ImageDraw


def draw_boxes_on_image(image_path, label_path, class_names):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    W, H = image.size

    colors = ['#FF3B30', '#34C759', '#007AFF', '#FF9500', '#AF52DE', '#5AC8FA']

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                class_id = int(parts[0])
                x_center, y_center, w, h = [float(v) for v in parts[1:5]]

                # Convert normalized YOLO coords to pixel corners
                x_center_px = x_center * W
                y_center_px = y_center * H
                w_px = w * W
                h_px = h * H

                x0 = x_center_px - w_px / 2
                y0 = y_center_px - h_px / 2
                x1 = x_center_px + w_px / 2
                y1 = y_center_px + h_px / 2

                color = colors[class_id % len(colors)]
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

                label_text = class_names.get(class_id, str(class_id))
                draw.text((x0 + 2, max(0, y0 - 14)), label_text, fill=color)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
        help='Path to YOLO dataset dir containing images/, labels/, data.yaml')
    parser.add_argument('--split', type=str, default='train',
        choices=['train', 'val'])
    parser.add_argument('--n_samples', type=int, default=8,
        help='Number of random frames to sample and visualize')
    parser.add_argument('--output_dir', type=str, default='spot_check_output',
        help='Directory to save annotated sample images to')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset_dir, 'images', args.split)
    labels_dir = os.path.join(args.dataset_dir, 'labels', args.split)
    data_yaml_path = os.path.join(args.dataset_dir, 'data.yaml')

    class_names = read_yaml_class_names(data_yaml_path)
    print('Classes: {}'.format(class_names))

    all_images = sorted([
        f for f in os.listdir(images_dir) if f.lower().endswith('.png')
    ])

    # Only sample from frames that actually have at least one labeled box,
    # otherwise you might randomly get an empty/negative frame and think
    # something's broken
    frames_with_labels = []
    for fname in all_images:
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + '.txt')
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            frames_with_labels.append(fname)

    print('{} / {} frames have at least one labeled box'.format(
        len(frames_with_labels), len(all_images)))

    random.seed(args.seed)
    n_samples = min(args.n_samples, len(frames_with_labels))
    sample_fnames = random.sample(frames_with_labels, n_samples)

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in sample_fnames:
        stem = os.path.splitext(fname)[0]
        image_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, stem + '.txt')

        annotated = draw_boxes_on_image(image_path, label_path, class_names)

        out_path = os.path.join(args.output_dir, 'spotcheck_' + fname)
        annotated.save(out_path)
        print('Saved {}'.format(out_path))

    print('\nDone. {} annotated sample images in {}'.format(
        n_samples, args.output_dir))


if __name__ == '__main__':
    main()