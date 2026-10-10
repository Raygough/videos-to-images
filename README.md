# videos-to-images

A pipeline for turning broadcast baseball footage into per-frame training data for player tracking research: extract frames, detect and filter players, map every frame back to its source video, and estimate monocular depth. Built to run as batch jobs on an HPC cluster (Slurm).

## Pipeline

```
video ──► extraction/ ──► detector/ ──► frame mapping ──► depth/
          (frames)        (YOLO boxes,   (frame ↔ video    (UniDepth maps
                           player filter)  provenance)       + intrinsics)
```

| Stage | Folder | What it does |
|-------|--------|--------------|
| 1. Extraction | `extraction/` | Decodes videos with OpenCV and writes image frames. |
| 2. Detection | `detector/` | Runs a YOLO detector on frames, keeps the player class, and visualizes labels. |
| 3. Frame mapping | `detector/` | Builds an index that maps each extracted frame back to its source video and frame number, and verifies it. |
| 4. Depth | `depth/` | Runs [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) over frames to produce depth maps and camera intrinsics. |

## Repository layout

```
videos-to-images/
├── extraction/   # video -> frames
├── detector/
│   ├── create_detector_env.sh          # environment setup
│   ├── detection.sh                    # run YOLO detection (batch job)
│   ├── filter.sh, filter_player_class.py   # keep only the player class
│   ├── build_frame_hash_index.py       # hash frames for matching
│   ├── build_full_frame_mapping*.py/.sh    # frame -> source video/frame mapping
│   ├── frame_index.sh
│   ├── match_job_to_sequence.py        # tie cluster jobs to video sequences
│   ├── verify_frame_mapping.py         # sanity-check the mapping
│   ├── visualize.sh, visualize_labels.py   # draw labels for spot checks
│   ├── back_proj.py                    # TODO: one-line description
│   ├── percentiles.py                  # TODO: one-line description
│   └── dataset.py
└── depth/
    ├── uni_loop.py     # run UniDepth over a set of frames
    ├── depth.sh        # batch job
    └── dataset.py
```

## Setup

```bash
git clone https://github.com/Raygough/videos-to-images.git
cd videos-to-images

# Detector environment
bash detector/create_detector_env.sh

# Depth: UniDepth is not vendored here; install it separately
git clone https://github.com/lpiccinelli-eth/UniDepth.git
pip install -e UniDepth
```

Model weights are not committed. Download the YOLO weights you want (for example `yolov8s.pt` from Ultralytics) and place them in `detector/`.

## Usage

The `.sh` scripts are Slurm batch scripts written for Yale's Bouchet cluster. Edit the paths, partition, and account fields at the top of each script before submitting.

```bash
# 1. Extract frames
# TODO: exact command for extraction/

# 2. Detect players
sbatch detector/detection.sh
sbatch detector/filter.sh

# 3. Build and verify the frame mapping
sbatch detector/frame_index.sh
sbatch detector/build_full_frame_mapping.sh
python detector/verify_frame_mapping.py   # TODO: arguments

# 4. Depth maps
sbatch depth/depth.sh
```

## Outputs

Generated data (frames, labels, depth maps, logs) is gitignored. Depth maps and logs live in `depth/depth_maps/` and `*.log` locally.

## Acknowledgments

- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) for monocular metric depth
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection

## License

MIT, see [LICENSE](LICENSE).
