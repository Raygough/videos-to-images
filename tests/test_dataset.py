"""Tests for detector/dataset.py and detector/filter_player_class.py."""

import os
import zipfile

import pytest
from dataset import copy_job_frames, extract_job_id, extract_zip, find_data_yaml_class_names, main
from filter_player_class import filter_split

# --- dataset: extract_job_id ------------------------------------------------

def test_extract_job_id_matches():
    assert extract_job_id("dataset_job_3844683.zip") == "3844683"


def test_extract_job_id_no_match_raises():
    with pytest.raises(ValueError):
        extract_job_id("plain.zip")


# --- dataset: find_data_yaml_class_names ------------------------------------

@pytest.fixture
def extracted_job(tmp_path):
    job = tmp_path / "job_1"
    imgs = job / "images" / "train"
    labels = job / "labels" / "train"
    imgs.mkdir(parents=True)
    labels.mkdir(parents=True)
    (job / "data.yaml").write_text(
        "path: .\nnames:\n  0: Home\n  1: First\n  4: Player\n"
    )
    return job


def test_find_data_yaml_class_names(extracted_job):
    names = find_data_yaml_class_names(str(extracted_job))
    assert names == {0: "Home", 1: "First", 4: "Player"}


def test_find_data_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_data_yaml_class_names(str(tmp_path))


# --- dataset: copy_job_frames ------------------------------------------------

def test_copy_job_frames_copies_and_prefixes(tmp_path):
    extract_to = tmp_path / "ex"
    imgs = extract_to / "images" / "train"
    labels = extract_to / "labels" / "train"
    imgs.mkdir(parents=True)
    labels.mkdir(parents=True)
    (imgs / "frame_000000.png").write_bytes(b"img")
    (labels / "frame_000000.txt").write_text("4 0.5 0.5 0.1 0.1 5 0.5 0.5 0.1 0.1\n")

    img_out = tmp_path / "i"
    lab_out = tmp_path / "l"
    img_out.mkdir()
    lab_out.mkdir()

    n = copy_job_frames("7", str(extract_to), str(img_out), str(lab_out))
    assert n == 1
    assert os.path.exists(img_out / "job7_frame_000000.png")
    assert os.path.exists(lab_out / "job7_frame_000000.txt")


def test_copy_job_frames_writes_empty_label_for_unlabeled(tmp_path):
    extract_to = tmp_path / "ex"
    imgs = extract_to / "images" / "train"
    labels = extract_to / "labels" / "train"
    imgs.mkdir(parents=True)
    labels.mkdir(parents=True)
    (imgs / "frame_000001.png").write_bytes(b"img")
    # no label file

    img_out = tmp_path / "i"
    lab_out = tmp_path / "l"
    img_out.mkdir()
    lab_out.mkdir()

    copy_job_frames("7", str(extract_to), str(img_out), str(lab_out))
    assert os.path.getsize(lab_out / "job7_frame_000001.txt") == 0


def test_copy_job_frames_missing_images_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        copy_job_frames("1", str(tmp_path), str(tmp_path), str(tmp_path))


# --- dataset: extract_zip ----------------------------------------------------

def test_extract_zip_writes_contents(tmp_path):
    zpath = tmp_path / "job.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("data.yaml", "names:\n  0: Home\n")
    dest = tmp_path / "dest"
    extract_zip(str(zpath), str(dest))
    assert (dest / "data.yaml").exists()


# --- dataset: job-level split writes train.txt/val.txt/data.yaml -------------

def _zip_job(zpath, job_id, class_id):
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("data.yaml", f"names:\n  {class_id}: Home\n")
        zf.writestr(f"images/train/frame_{job_id}_000000.png", b"img")
        zf.writestr(f"labels/train/frame_{job_id}_000000.txt", f"{class_id} 0.5 0.5 0.1 0.1\n")


def test_build_dataset_pipeline(tmp_path, monkeypatch):
    zip_dir = tmp_path / "zips"
    zip_dir.mkdir()
    out_dir = tmp_path / "dataset"
    for jid in ("1", "2", "3"):
        _zip_job(zip_dir / f"dataset_job_{jid}.zip", jid, 4)

    monkeypatch.setattr("sys.argv", [
        "dataset.py",
        "--zip_dir", str(zip_dir),
        "--output_dir", str(out_dir),
        "--val_fraction", "0.3",
        "--seed", "1",
    ])
    main()

    assert (out_dir / "data.yaml").exists()
    assert (out_dir / "train.txt").exists()
    assert (out_dir / "val.txt").exists()
    assert " 4: Home" in (out_dir / "data.yaml").read_text()
    # split set output dirs were created
    assert (out_dir / "images" / "train").is_dir()
    assert (out_dir / "images" / "val").is_dir()


def test_build_dataset_no_zips_raises(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "dataset"
    monkeypatch.setattr("sys.argv", [
        "dataset.py",
        "--zip_dir", str(empty),
        "--output_dir", str(out),
    ])
    with pytest.raises(ValueError):
        main()


# --- filter_player_class: filter_split ----------------------------------------

def test_filter_split_remaps_player_and_drops_others(tmp_path):
    imgs_in = tmp_path / "imgs_in"; lab_in = tmp_path / "lab_in"
    imgs_out = tmp_path / "imgs_out"; lab_out = tmp_path / "lab_out"
    for d in (imgs_in, lab_in):
        d.mkdir(parents=True)

    (imgs_in / "f.png").write_bytes(b"img")
    (lab_in / "f.txt").write_text(
        "4 0.5 0.5 0.1 0.1\n"   # Player -> keep, remap to 0
        "0 0.1 0.1 0.1 0.1\n"   # Home -> drop
        "5 0.9 0.9 0.1 0.1\n"   # Ignore -> drop
    )

    n_frames, n_pos, n_boxes = filter_split(
        str(imgs_in), str(lab_in), str(imgs_out), str(lab_out), player_class_id=4
    )

    assert n_frames == 1 and n_pos == 1 and n_boxes == 1
    kept = (lab_out / "f.txt").read_text().strip()
    assert kept == "0 0.5 0.5 0.1 0.1"


def test_filter_split_empty_label_writes_empty_file(tmp_path):
    imgs_in = tmp_path / "i"; lab_in = tmp_path / "l"
    imgs_out = tmp_path / "io"; lab_out = tmp_path / "lo"
    for d in (imgs_in, lab_in):
        d.mkdir(parents=True)
    (imgs_in / "f.png").write_bytes(b"img")

    filter_split(str(imgs_in), str(lab_in), str(imgs_out), str(lab_out), 4)
    assert (lab_out / "f.txt").exists()
    assert (lab_out / "f.txt").read_text() == ""


def test_filter_split_skips_non_png(tmp_path):
    imgs_in = tmp_path / "i"; lab_in = tmp_path / "l"
    imgs_out = tmp_path / "io"; lab_out = tmp_path / "lo"
    for d in (imgs_in, lab_in):
        d.mkdir(parents=True)
    (imgs_in / "notes.txt").write_text("x")
    n_frames, n_pos, n_boxes = filter_split(
        str(imgs_in), str(lab_in), str(imgs_out), str(lab_out), 4
    )
    assert n_frames == 0 and n_pos == 0 and n_boxes == 0
    assert os.listdir(imgs_out) == []