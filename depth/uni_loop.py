import os
import json
import argparse

import numpy as np
import torch
from PIL import Image

from unidepth.models import UniDepthV2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--frames_root',
        type=str, required=True,
        help='Root directory containing per-sequence subfolders of frames')
    parser.add_argument('--output_root',
        type=str, required=True,
        help='Root directory to write depth maps and intrinsics to')
    parser.add_argument('--model_name',
        type=str, default='lpiccinelli/unidepth-v2-vitl14',
        help='Hugging Face model id to load, e.g. lpiccinelli/unidepth-v2-vitl14')
    parser.add_argument('--device',
        type=str, default='cuda', help='Device to run on: cuda or cpu')
    parser.add_argument('--sequences',
    type=str, nargs='*', default=None,
    help='Optional list of sequence subfolder names to process; '
         'if omitted, all subfolders under frames_root are processed')

    return parser.parse_args()


def get_sequence_dirs(frames_root, sequences=None):
    '''
    Get list of sequence subdirectories to process

    Arg(s):
        frames_root : str
            root directory containing per-sequence subfolders
    Returns:
        list[str] : sorted list of sequence directory names
    '''

    #gets all videos in frames directory
    all_entries = sorted(os.listdir(frames_root))

    #gets each video name and puts it in sequence_dirs 
    #i.e. [4125352, 25081729, ...]
    sequence_dirs = [
        entry for entry in all_entries
        if os.path.isdir(os.path.join(frames_root, entry))
    ]

    if sequences is not None:
        sequence_dirs = [s for s in sequence_dirs if s in sequences]

    return sequence_dirs

def get_frame_paths(sequence_path):
    '''
    Get sorted list of frame image paths within a sequence directory
 
    Arg(s):
        sequence_path : str
            path to a single sequence directory of frames
    Returns:
        list[str] : sorted list of frame filenames (not full paths)
    '''
 
    frame_files = sorted([
        f for f in os.listdir(sequence_path)
        if f.lower().endswith('.jpg')
    ])
 
    return frame_files


def load_model(model_name, device):
    '''
    Load UniDepthV2 model from Hugging Face and move to device

    Arg(s):
        model_name : str
            Hugging Face model id
        device : torch.device
            device to move model to
    Returns:
        UniDepthV2 : model in eval mode on device
    '''

    model = UniDepthV2.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    return model


def load_image_as_tensor(image_path):
    '''
    Load an RGB image from disk as a C x H x W uint8 tensor

    Arg(s):
        image_path : str
            path to image file
    Returns:
        torch.Tensor[uint8] : C x H x W image tensor
    '''

    #opening with pillow and converting to rgb
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    return image_tensor


def run_inference(model, rgb_tensor, device):
    '''
    Run UniDepth inference on a single RGB image tensor

    Arg(s):
        model : UniDepthV2
            loaded model
        rgb_tensor : torch.Tensor[uint8]
            C x H x W image tensor
        device : torch.device
            device to run inference on
    Returns:
        np.ndarray[float32] : H x W depth map
        np.ndarray[float32] : 3 x 3 intrinsics matrix
    '''

    rgb_tensor = rgb_tensor.to(device)

    with torch.no_grad():
        #unidepth runs inference on single tensor
        predictions = model.infer(rgb_tensor)

    #extract depth map and intrinsics from single tensor
    depth = predictions['depth']
    intrinsics = predictions['intrinsics']

    # Squeeze any leading batch dimension if present
    depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
    intrinsics = intrinsics.squeeze().detach().cpu().numpy().astype(np.float32)

    return depth, intrinsics


def process_sequence(model, frames_root, output_root, sequence_name, device):
    '''
    Process all frames within a single sequence directory

    Arg(s):
        model : UniDepthV2
            loaded model
        frames_root : str
            root directory containing sequence subfolders
        output_root : str
            root directory to write outputs to
        sequence_name : str
            name of sequence subfolder to process
        device : torch.device
            device to run inference on
    '''

    sequence_input_path = os.path.join(frames_root, sequence_name)
    sequence_output_path = os.path.join(output_root, sequence_name)

    os.makedirs(sequence_output_path, exist_ok=True)

    frame_files = get_frame_paths(sequence_input_path)

    print('Processing video: {} ({} frames)'.format(sequence_name, len(frame_files)))

    for frame_file in frame_files:

        frame_id = os.path.splitext(frame_file)[0]

        depth_output_path = os.path.join(
            sequence_output_path, '{}_depth.npy'.format(frame_id))
        intrinsics_output_path = os.path.join(
            sequence_output_path, '{}_intrinsics.json'.format(frame_id))

        #skip frames that have already been processed
        if os.path.exists(depth_output_path) and os.path.exists(intrinsics_output_path):
            continue

        frame_path = os.path.join(sequence_input_path, frame_file)

        try:
            rgb_tensor = load_image_as_tensor(frame_path)
            depth, intrinsics = run_inference(model, rgb_tensor, device)

            np.save(depth_output_path, depth)

            with open(intrinsics_output_path, 'w') as f:
                json.dump({'intrinsics': intrinsics.tolist()}, f)

        except Exception as e:
            print('WARNING: failed on frame {} in video {}: {}'.format(
                frame_file, sequence_name, e))
            continue

    print('Finished video: {}'.format(sequence_name))


if __name__ == '__main__':

    args = parse_args()

    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    print('Loading model {} on {}...'.format(args.model_name, device))
    model = load_model(args.model_name, device)

    os.makedirs(args.output_root, exist_ok=True)

    sequence_dirs = get_sequence_dirs(args.frames_root, args.sequences)

    print('Found {} videos to process'.format(len(sequence_dirs)))

    for sequence_name in sequence_dirs:
        process_sequence(
            model=model,
            frames_root=args.frames_root,
            output_root=args.output_root,
            sequence_name=sequence_name,
            device=device
        )

    print('All videos processed. Output written to: {}'.format(args.output_root))