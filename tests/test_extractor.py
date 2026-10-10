"""Tests for extractor.py — the shared frame-extraction engine.

The headline regression is the old hardcoded ``videos[342:]`` slice in
``extraction/main.py``: it silently skipped the first 342 videos. The engine
must process every video, support step/skip sampling, and report errors
cleanly instead of exiting mid-batch.
"""

import os

import pytest

from extractor import (
    build_parser,
    extract_directory,
    extract_video,
    run_cli,
)

# --- extract_video -----------------------------------------------------------

def test_extract_video_writes_padded_frames(tiny_video, tmp_path):
    out = str(tmp_path / "out")
    result = extract_video(tiny_video, out)
    assert result.ok
    assert result.frames_written == 12
    assert result.output_dir == os.path.join(out, "clip")
    files = sorted(os.listdir(result.output_dir))
    assert files == [f"{i:07d}.jpg" for i in range(12)]


def test_extract_video_step_keeps_every_nth(tiny_video, tmp_path):
    result = extract_video(tiny_video, str(tmp_path / "o"), step=3)
    assert result.frames_written == 4  # 0,3,6,9
    files = sorted(os.listdir(result.output_dir))
    assert files == ["0000000.jpg", "0000003.jpg", "0000006.jpg", "0000009.jpg"]


def test_extract_video_skip_drops_leading_frames(tiny_video, tmp_path):
    result = extract_video(tiny_video, str(tmp_path / "o"), skip=2)
    assert result.frames_written == 10
    first = result.output_dir and sorted(os.listdir(result.output_dir))[0]
    assert first == "0000002.jpg"  # index 2 is the first frame kept


def test_extract_video_missing_file_not_ok(tmp_path):
    result = extract_video(str(tmp_path / "nope.mp4"), str(tmp_path / "o"))
    assert not result.ok
    assert result.error is not None


def test_extract_video_custom_pad_and_ext(tiny_video, tmp_path):
    result = extract_video(tiny_video, str(tmp_path / "o"), pad=3, ext=".png")
    assert result.frames_written == 12
    assert sorted(os.listdir(result.output_dir))[0] == "000.png"


def test_extract_video_stem_used_for_dir(tiny_video, tmp_path):
    r = extract_video(tiny_video, str(tmp_path / "o"))
    assert "clip" in r.output_dir


# --- extract_directory -------------------------------------------------------

def test_extract_directory_returns_results_and_total(tiny_video, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "clip.mp4").symlink_to(tiny_video)
    results, total = extract_directory(str(src), str(tmp_path / "o"))
    assert len(results) == 1
    assert results[0].ok
    assert total == 12


def test_extract_directory_honors_only_extensions(tiny_video, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "clip.mp4").symlink_to(tiny_video)
    results, total = extract_directory(
        str(src), str(tmp_path / "o"), only_extensions=(".txt",)
    )
    assert len(results) == 0
    assert total == 0


def test_extract_directory_processes_all_videos(tiny_video, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "clip.mp4").symlink_to(tiny_video)
    (src / "clip2.mp4").symlink_to(tiny_video)
    results, total = extract_directory(str(src), str(tmp_path / "o"))
    assert len(results) == 2
    assert total == 24


def test_extract_directory_raises_for_non_dir(tmp_path):
    with pytest.raises(NotADirectoryError):
        extract_directory(str(tmp_path / "missing"), str(tmp_path / "o"))


# --- CLI (regression: the old 342-slice) -------------------------------------

def test_run_cli_processes_all_videos_no_hidden_offset(tiny_video, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "clip.mp4").symlink_to(tiny_video)
    outdir = str(tmp_path / "o")
    code = run_cli([str(src), outdir])
    assert code == 0
    # Every video produced frames; nothing skipped.
    assert sorted(os.listdir(outdir)) == ["clip"]


def test_run_cli_step_flag(tiny_video, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "clip.mp4").symlink_to(tiny_video)
    outdir = str(tmp_path / "o")
    code = run_cli([str(src), outdir, "--step", "4"])
    assert code == 0
    clip_dir = os.path.join(outdir, "clip")
    assert len(os.listdir(clip_dir)) == 3  # 0,4,8


def test_run_cli_missing_dir_returns_1(tmp_path):
    assert run_cli([str(tmp_path / "missing"), str(tmp_path / "o")]) == 1


def test_run_cli_invalid_step_rejected(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(SystemExit):
        run_cli([str(src), str(tmp_path / "o"), "--step", "0"])


def test_run_cli_nonzero_when_all_videos_fail(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "broken.mp4").write_bytes(b"not a real video")
    code = run_cli([str(src), str(tmp_path / "o")])
    assert code == 1  # every video failed -> failure


def test_run_cli_zero_when_one_video_succeeds(tiny_video, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "broken.mp4").write_bytes(b"not a real video")
    (src / "clip.mp4").symlink_to(tiny_video)
    code = run_cli([str(src), str(tmp_path / "o")])
    assert code == 0  # partial success is not a failure
    # The readable video extracted frames; the broken one left an empty dir.
    assert len(os.listdir(os.path.join(str(tmp_path / "o"), "clip"))) == 12
    assert os.listdir(os.path.join(str(tmp_path / "o"), "broken")) == []


def test_build_parser_default_step_is_one():
    p = build_parser()
    ns = p.parse_args(["a", "b"])
    assert ns.step == 1 and ns.skip == 0