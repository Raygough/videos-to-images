"""Shared frame-extraction engine for videos-to-images.

The original ``main.py`` / ``extraction/main.py`` each ran a copy of the same
loop, and `extraction/main.py` carried a hardcoded ``videos = videos[342:]``
slice that silently skipped the first 342 input videos. Both now delegate to
the single engine below, which:

* processes every video (no hidden offset),
* skips unreadable videos instead of aborting the whole batch,
* supports a contiguous ``--step``/``--skip`` sampling mode and frame name
  padding and extension overrides,
* returns per-video and aggregate statistics, and exits non-zero on failure.

Requires OpenCV (``cv2``).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

__all__ = ["extract_video", "extract_directory", "build_parser", "run_cli"]


@dataclass
class VideoResult:
    """Outcome of extracting frames from one video."""

    video: str
    frames_written: int = 0
    ok: bool = True
    error: str | None = None
    output_dir: str = ""

    def __post_init__(self) -> None:
        self.video = os.path.basename(self.video)


def _safe_stem(path: str) -> str:
    """Return the filename without its extension, guarding empty names."""
    return os.path.splitext(os.path.basename(path))[0] or "video"


def extract_video(
    video_path: str,
    output_path: str,
    *,
    pad: int = 7,
    ext: str = ".jpg",
    step: int = 1,
    skip: int = 0,
) -> VideoResult:
    """Extract frames from a single video into a per-video output directory.

    Frames are named ``{index:0{pad}d}{ext}``. ``step`` keeps every Nth frame;
    ``skip`` drops the first ``skip`` frames (useful for a preview where the
    same clip is sampled at increasing spacing). Returns a :class:`VideoResult`
    that is ``ok=False`` with a message if the video cannot be opened.

    Args:
        video_path: Path to a video file.
        output_path: Directory under which a subdir (named after the video
            stem) is created to hold the frames.
        pad: Zero-padding width for frame filenames.
        ext: File extension for saved frames (must include the leading dot).
        step: Save every ``step``-th frame (``1`` = every frame).
        skip: Number of initial frames to discard before saving.
    """
    import cv2  # local import keeps the module usable without OpenCV

    stem = _safe_stem(video_path)
    output_dir = os.path.join(output_path, stem)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    result = VideoResult(video=video_path, output_dir=output_dir)
    if not cap.isOpened():
        result.ok = False
        result.error = "cannot open video"
        return result

    try:
        idx = 0
        written = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if idx >= skip and (idx - skip) % step == 0:
                frame_name = f"{idx:0{pad}d}{ext}"
                dst = os.path.join(output_dir, frame_name)
                if not cv2.imwrite(dst, frame):
                    result.ok = False
                    result.error = f"write failed for {frame_name}"
                    break
                written += 1
            idx += 1
    finally:
        cap.release()

    result.frames_written = written
    return result


def extract_directory(
    video_path: str,
    output_path: str,
    *,
    pad: int = 7,
    ext: str = ".jpg",
    step: int = 1,
    skip: int = 0,
    only_extensions: tuple[str, ...] = (),
    verbose: bool = False,
) -> tuple[list[VideoResult], int]:
    """Extract frames from every video in ``video_path``.

    Args:
        video_path: Directory of source videos.
        output_path: Directory to write per-video frame subdirectories into.
        pad, ext, step, skip: Forwarded to :func:`extract_video`.
        only_extensions: If non-empty, only process files with one of these
            (case-insensitive) extensions.
        verbose: Print a line per processed video.

    Returns:
        ``(results, total_frames)`` where ``results`` holds one
        :class:`VideoResult` per input file examined.

    Raises:
        NotADirectoryError: If ``video_path`` is not an existing directory.
    """
    if not os.path.isdir(video_path):
        raise NotADirectoryError(f"{video_path} is not a directory")
    os.makedirs(output_path, exist_ok=True)

    only = {e.lower().lstrip(".") for e in only_extensions}
    results: list[VideoResult] = []
    total_frames = 0

    for name in sorted(os.listdir(video_path)):
        full = os.path.join(video_path, name)
        if only and not name.lower().endswith(tuple("." + e for e in only)):
            continue
        if not os.path.isfile(full):
            continue

        r = extract_video(full, output_path, pad=pad, ext=ext, step=step, skip=skip)
        results.append(r)
        total_frames += r.frames_written
        if verbose:
            status = "ok" if r.ok else f"ERROR: {r.error}"
            print(f"{name}: {r.frames_written} frames -> {status}")

    return results, total_frames


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by main.py and extraction/main.py."""
    parser = argparse.ArgumentParser(
        prog="videos-to-images",
        description="Extract image frames from videos with OpenCV.",
    )
    parser.add_argument("videopath", help="Directory containing video files")
    parser.add_argument("outputpath", help="Directory to write extracted frames to")
    parser.add_argument("--step", type=int, default=1,
                        help="Save every Nth frame (default: every frame)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Number of initial frames to discard (default: 0)")
    parser.add_argument("--pad", type=int, default=7,
                        help="Zero-padding width for frame names (default: 7)")
    parser.add_argument("--ext", default=".jpg",
                        help="Frame file extension incl. dot (default: .jpg)")
    parser.add_argument("--only-extensions", default="",
                        help="Comma-separated video extensions to process, e.g. "
                             ".mp4,.avi (default: all files)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a line for every processed video")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Returns:
        ``0`` on success (even if some videos were skipped, as long as at least
        one was processed), ``1`` if every video failed or the source dir is
        invalid, ``2`` for a usage error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.step < 1:
        parser.error("--step must be >= 1")
    if args.skip < 0:
        parser.error("--skip must be >= 0")
    if not args.ext.startswith("."):
        parser.error("--ext must start with a dot")

    if not os.path.isdir(args.videopath):
        print(f"Error: {args.videopath} is not a directory", file=sys.stderr)
        return 1

    only_exts = tuple(e.strip() for e in args.only_extensions.split(",") if e.strip())
    try:
        results, total = extract_directory(
            args.videopath,
            args.outputpath,
            pad=args.pad,
            ext=args.ext,
            step=args.step,
            skip=args.skip,
            only_extensions=only_exts,
            verbose=args.verbose,
        )
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    failed = [r for r in results if not r.ok]
    n_ok = len(results) - len(failed)
    print(f"Processed {len(results)} files, {total} frames written")
    if failed:
        print(f"{len(failed)} file(s) could not be read:", file=sys.stderr)
        for r in failed:
            print(f"  {r.video}: {r.error}", file=sys.stderr)
        # Non-zero only if nothing at all was extracted; partial success is 0.
        return 1 if n_ok == 0 else 0
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())