import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.pipeline_mono_dataset import (
    CLASS_LABELS,
    SegmentInferenceResult,
    derive_masks_from_semantic,
    parse_args,
    run_pipeline,
    run_segment_pipeline,
)


def _write_image(path: Path) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_segment(segment_dir: Path) -> None:
    (segment_dir / "images").mkdir(parents=True)
    for stem in ("1000", "2000"):
        _write_image(segment_dir / "images" / f"{stem}.jpeg")
    (segment_dir / "timestamps.txt").write_text("1000\n2000\n", encoding="utf-8")
    (segment_dir / "camera_params.json").write_text(
        json.dumps(
            {
                "num_cameras": 2,
                "extrinsics": [],
                "intrinsics": [
                    {"camera_id": "1000", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                    {"camera_id": "2000", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (segment_dir.parent / "source_calib.json").write_text(
        json.dumps(
            {
                "camera_name": "mid_center_top_wide",
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                "distortion": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
            }
        ),
        encoding="utf-8",
    )


def test_derive_masks_from_semantic_writes_expected_binary_conventions(tmp_path: Path) -> None:
    semantic = np.array(
        [
            [16, 0, 0, 31],
            [16, 1, 2, 31],
            [3, 3, 3, 3],
        ],
        dtype=np.uint8,
    )

    outputs = derive_masks_from_semantic(semantic, edge_dilation=0)

    assert set(outputs) == {
        "sky_mask",
        "ego_mask",
        "semantic_edge_mask",
        "worldmirror_keep_mask",
    }
    assert outputs["sky_mask"].tolist() == [
        [255, 0, 0, 0],
        [255, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert outputs["ego_mask"].tolist() == [
        [0, 0, 0, 255],
        [0, 0, 0, 255],
        [0, 0, 0, 0],
    ]
    assert outputs["semantic_edge_mask"][1, 1] == 255
    assert outputs["worldmirror_keep_mask"][0, 0] == 0
    assert outputs["worldmirror_keep_mask"][0, 3] == 0
    assert set(np.unique(outputs["worldmirror_keep_mask"])) <= {0, 255}


def test_run_segment_pipeline_writes_segment_outputs_with_fake_segmenter(tmp_path: Path) -> None:
    segment_dir = tmp_path / "seg_000"
    _write_segment(segment_dir)

    class FakeSegmenter:
        def infer(self, images):
            masks = []
            for index, image in enumerate(images):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[:, 0] = 16
                if index == 1:
                    mask[:, -1] = 31
                masks.append(mask)
            return masks

    result = run_segment_pipeline(
        segment_dir=segment_dir,
        source_calib_path=segment_dir.parent / "source_calib.json",
        model_path=Path("/tmp/fake-mask2former.pt"),
        segmenter=FakeSegmenter(),
        batch_size=2,
        save_color_viz=True,
        overwrite=True,
    )

    assert result == SegmentInferenceResult(
        segment_id="seg_000",
        planned_frames=2,
        written_semantic=2,
        written_masks=2,
        dry_run=False,
    )
    assert (segment_dir / "image_segment" / "semantic" / "1000.png").exists()
    assert (segment_dir / "image_segment" / "undistorted" / "1000.jpeg").exists()
    assert (segment_dir / "image_segment" / "color_viz" / "1000.png").exists()
    assert (segment_dir / "masks" / "sky_mask" / "1000.png").exists()
    assert (segment_dir / "masks" / "ego_mask" / "2000.png").exists()
    assert (segment_dir / "masks" / "worldmirror_keep_mask" / "2000.png").exists()

    class_labels = json.loads((segment_dir / "image_segment" / "class_labels.json").read_text())
    assert class_labels["16"] == CLASS_LABELS[16] == "sky"
    assert class_labels["31"] == CLASS_LABELS[31] == "ego_mask"

    image_manifest = json.loads((segment_dir / "image_segment" / "manifest.json").read_text())
    assert image_manifest["pipeline"] == "mask2former_segment_v1"
    assert image_manifest["calibration_source"] == "../source_calib.json"
    assert image_manifest["frames"][0]["timestamp_ns"] == "1000"

    masks_manifest = json.loads((segment_dir / "masks" / "manifest.json").read_text())
    assert masks_manifest["pixel_conventions"]["sky_mask"] == "255=sky, 0=non-sky"
    assert masks_manifest["class_ids"] == {"sky": 16, "ego_mask": 31}


def test_pipeline_script_help_runs_when_invoked_by_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pipeline_mono_dataset.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--skip-build-dataset" in result.stdout


def test_run_pipeline_dry_run_reuses_existing_dataset_track(tmp_path: Path) -> None:
    track_dir = tmp_path / "dataset" / "tracks" / "644" / "mid_center_top_wide"
    segment_dir = track_dir / "seg_000"
    _write_segment(segment_dir)

    args = parse_args(
        [
            "--skip-build-dataset",
            "--dataset-track-dir",
            str(track_dir),
            "--dry-run",
            "--model-path",
            str(tmp_path / "missing-model.pt"),
        ]
    )

    manifest = run_pipeline(args)

    assert manifest["dry_run"] is True
    assert manifest["segment_count"] == 1
    assert manifest["segments"][0]["planned_frames"] == 2
    assert (track_dir / "mask_generation_manifest.json").exists()
    image_manifest = json.loads((segment_dir / "image_segment" / "manifest.json").read_text())
    assert image_manifest["dry_run"] is True
    masks_manifest = json.loads((segment_dir / "masks" / "manifest.json").read_text())
    assert masks_manifest["dry_run"] is True
