import json
from pathlib import Path

import pytest

from scripts.make_mono_dataset import (
    build_intrinsics_only_camera_params,
    build_qc_report,
    extract_camera_calib,
    match_poses_to_images,
    renumber_valid_segments,
    split_segments,
)


def _write_image(path: Path) -> None:
    path.write_bytes(b"\xff\xd8\xff\xd9")


def _sample_calib() -> dict:
    return {
        "cameras": [
            {
                "name": {"label": "SENSOR_LABEL_REAR_CENTER_TOP_NORM"},
                "width": 1920,
                "height": 1080,
                "specific_model": {"model": "CAMERA_MODEL_PINHOLE"},
            },
            {
                "name": {"label": "SENSOR_LABEL_MID_CENTER_TOP_WIDE"},
                "width": 3840,
                "height": 2160,
                "specific_model": {
                    "model": "CAMERA_MODEL_KANNEL_BRAND",
                    "kb_model": {
                        "fu": 1904.0,
                        "fv": 1903.0,
                        "pu": 1917.0,
                        "pv": 1076.0,
                        "distortions_k1": -0.01,
                        "distortions_k2": -0.02,
                        "distortions_k3": -0.03,
                        "distortions_k4": -0.04,
                    },
                },
                "sensor_to_vehicle": {"translation_x_m": 1.0},
                "sensor_to_vehicle_eol": {"translation_x_m": 2.0},
                "serial_number": "camera-serial",
                "status": "CALIB_STATUS_ONLINE_SELF_CALIBRATED",
            },
        ]
    }


def test_extract_camera_calib_normalizes_mid_center_top_wide_fields() -> None:
    calib = extract_camera_calib(_sample_calib(), "mid_center_top_wide")

    assert calib["camera_name"] == "mid_center_top_wide"
    assert calib["sensor_label"] == "SENSOR_LABEL_MID_CENTER_TOP_WIDE"
    assert calib["image_width"] == 3840
    assert calib["image_height"] == 2160
    assert calib["camera_model"] == "CAMERA_MODEL_KANNEL_BRAND"
    assert calib["intrinsics"] == {
        "fx": 1904.0,
        "fy": 1903.0,
        "cx": 1917.0,
        "cy": 1076.0,
    }
    assert calib["distortion"] == {
        "k1": -0.01,
        "k2": -0.02,
        "k3": -0.03,
        "k4": -0.04,
    }


def test_match_poses_to_images_requires_exact_timestamp_match(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "1000.jpeg")
    _write_image(image_dir / "2000.jpeg")
    poses = [
        {"timestamp": 1000, "name": "1000", "pose": {"trans": [1, 2, 3]}},
        {"timestamp": 3000, "name": "3000", "pose": {"trans": [4, 5, 6]}},
    ]

    matches, report = match_poses_to_images(poses, image_dir)

    assert [m["timestamp"] for m in matches] == ["1000"]
    assert matches[0]["image_path"].endswith("1000.jpeg")
    assert report["exact_match_count"] == 1
    assert report["missing_pose_timestamps"] == ["3000"]


def test_split_segments_respects_gap_and_max_frames() -> None:
    matches = [
        {"timestamp": "1000000000"},
        {"timestamp": "1200000000"},
        {"timestamp": "1400000000"},
        {"timestamp": "3000000000"},
        {"timestamp": "3200000000"},
    ]

    report = split_segments(matches, max_gap_s=1.0, min_frames=2, max_frames=2)

    assert [s["segment_id"] for s in report["segments"]] == ["seg_000", "seg_001", "seg_002"]
    assert [s["frame_count"] for s in report["segments"]] == [2, 1, 2]
    assert report["dropped_segments"][0]["segment_id"] == "seg_001"
    assert report["dropped_segments"][0]["reason"] == "frame_count_less_than_min_frames"


def test_renumber_valid_segments_keeps_public_original_segment_id() -> None:
    split_report = {
        "segments": [
            {"segment_id": "seg_000", "frame_count": 1, "frames": []},
            {"segment_id": "seg_001", "frame_count": 3, "frames": []},
            {"segment_id": "seg_002", "frame_count": 1, "frames": []},
            {"segment_id": "seg_003", "frame_count": 4, "frames": []},
        ],
        "dropped_segments": [],
    }

    valid = renumber_valid_segments(split_report, min_frames=2)

    assert [s["segment_id"] for s in valid] == ["seg_000", "seg_001"]
    assert [s["source_segment_id"] for s in valid] == ["seg_001", "seg_003"]


def test_build_intrinsics_only_camera_params_covers_every_frame() -> None:
    segment = {
        "frames": [
            {"timestamp": "1000"},
            {"timestamp": "2000"},
        ]
    }
    calib = {
        "intrinsics": {
            "fx": 1904.0,
            "fy": 1903.0,
            "cx": 1917.0,
            "cy": 1076.0,
        }
    }

    camera_params = build_intrinsics_only_camera_params(segment, calib)

    assert camera_params["num_cameras"] == 2
    assert camera_params["extrinsics"] == []
    assert [item["camera_id"] for item in camera_params["intrinsics"]] == ["1000", "2000"]
    assert camera_params["intrinsics"][0]["matrix"] == [
        [1904.0, 0.0, 1917.0],
        [0.0, 1903.0, 1076.0],
        [0.0, 0.0, 1.0],
    ]


def test_build_qc_report_detects_missing_image_and_prior_mismatch(tmp_path: Path) -> None:
    segment_dir = tmp_path / "seg_000"
    image_dir = segment_dir / "images"
    image_dir.mkdir(parents=True)
    _write_image(image_dir / "1000.jpeg")
    segment = {
        "frame_count": 2,
        "max_gap_s": 1.5,
        "policy": {"max_gap_s": 1.0},
        "frames": [
            {"timestamp": "1000", "image_path": str(tmp_path / "src" / "1000.jpeg")},
            {"timestamp": "2000", "image_path": str(tmp_path / "src" / "2000.jpeg")},
        ],
    }
    camera_params = {
        "num_cameras": 2,
        "extrinsics": [],
        "intrinsics": [{"camera_id": "1000", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}],
    }

    qc = build_qc_report(segment, segment_dir, camera_params)

    assert qc["passed"] is False
    assert qc["checks"]["image_files_exist"] is False
    assert qc["checks"]["camera_id_match_intrinsics"] is False
    assert qc["checks"]["max_gap_s_within_policy"] is False
