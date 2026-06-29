#!/usr/bin/env python3
"""Build a mono WorldMirror inference dataset from one track directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

IMAGE_PATTERNS = ("*.jpeg", "*.jpg", "*.png", "*.webp")


def camera_name_to_sensor_label(camera_name: str) -> str:
    return f"SENSOR_LABEL_{camera_name.upper()}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_camera_calib(calib_data: dict[str, Any], camera_name: str) -> dict[str, Any]:
    target_label = camera_name_to_sensor_label(camera_name)
    camera = None
    for item in calib_data.get("cameras", []):
        if item.get("name", {}).get("label") == target_label:
            camera = item
            break
    if camera is None:
        raise ValueError(f"Camera label not found in calib.json: {target_label}")

    specific_model = camera.get("specific_model", {})
    camera_model = specific_model.get("model")
    kb_model = specific_model.get("kb_model")
    if not kb_model:
        raise ValueError(f"Camera {target_label} does not contain specific_model.kb_model")

    normalized = {
        "camera_name": camera_name,
        "sensor_label": target_label,
        "image_width": int(camera["width"]),
        "image_height": int(camera["height"]),
        "camera_model": camera_model,
        "intrinsics": {
            "fx": float(kb_model["fu"]),
            "fy": float(kb_model["fv"]),
            "cx": float(kb_model["pu"]),
            "cy": float(kb_model["pv"]),
        },
        "distortion": {
            "k1": float(kb_model.get("distortions_k1", 0.0)),
            "k2": float(kb_model.get("distortions_k2", 0.0)),
            "k3": float(kb_model.get("distortions_k3", 0.0)),
            "k4": float(kb_model.get("distortions_k4", 0.0)),
        },
        "sensor_to_vehicle": camera.get("sensor_to_vehicle"),
        "sensor_to_vehicle_eol": camera.get("sensor_to_vehicle_eol"),
        "serial_number": camera.get("serial_number"),
        "update_timestamp": camera.get("update_timestamp"),
        "status": camera.get("status"),
        "raw_camera": camera,
    }
    return normalized


def list_images(image_dir: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        images.extend(image_dir.glob(pattern))
    return sorted(images, key=lambda p: p.stem)


def load_image_index(image_dir: Path) -> dict[str, Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    image_by_stem = {p.stem: p for p in list_images(image_dir)}
    if not image_by_stem:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return image_by_stem


def _nearest_error_ns(timestamp: int, sorted_image_ts: list[int]) -> int | None:
    if not sorted_image_ts:
        return None
    lo, hi = 0, len(sorted_image_ts)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_image_ts[mid] < timestamp:
            lo = mid + 1
        else:
            hi = mid
    candidates = []
    if lo < len(sorted_image_ts):
        candidates.append(abs(sorted_image_ts[lo] - timestamp))
    if lo > 0:
        candidates.append(abs(sorted_image_ts[lo - 1] - timestamp))
    return min(candidates) if candidates else None


def match_poses_to_images(
    poses: list[dict[str, Any]], image_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_paths = list_images(image_dir)
    image_by_stem = {p.stem: p for p in image_paths}
    image_index = {p.stem: i for i, p in enumerate(image_paths)}
    sorted_image_ts = sorted(int(p.stem) for p in image_paths if p.stem.isdigit())

    matches: list[dict[str, Any]] = []
    missing: list[str] = []
    nearest_errors: list[int] = []

    for pose in poses:
        timestamp = str(pose.get("timestamp", pose.get("name")))
        name = str(pose.get("name", timestamp))
        stem = timestamp if timestamp in image_by_stem else name
        image_path = image_by_stem.get(stem)
        if image_path is None:
            missing.append(timestamp)
            if timestamp.isdigit():
                err = _nearest_error_ns(int(timestamp), sorted_image_ts)
                if err is not None:
                    nearest_errors.append(err)
            continue
        matches.append(
            {
                "timestamp": stem,
                "image_path": str(image_path),
                "image_index": image_index[stem],
                "pose": pose,
            }
        )
        nearest_errors.append(0)

    matched_indices = [m["image_index"] for m in matches]
    report = {
        "image_count": len(image_paths),
        "pose_count": len(poses),
        "exact_match_count": len(matches),
        "missing_pose_timestamps": missing,
        "nearest_error_ns": {
            "min": min(nearest_errors) if nearest_errors else None,
            "max": max(nearest_errors) if nearest_errors else None,
            "mean": mean(nearest_errors) if nearest_errors else None,
        },
        "first_matched_image_index": min(matched_indices) if matched_indices else None,
        "last_matched_image_index": max(matched_indices) if matched_indices else None,
    }
    return matches, report


def split_segments(
    matches: list[dict[str, Any]], max_gap_s: float, min_frames: int, max_frames: int
) -> dict[str, Any]:
    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    if max_frames < min_frames:
        raise ValueError("max_frames must be >= min_frames")

    ordered = sorted(matches, key=lambda m: int(m["timestamp"]))
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    max_gap_ns = int(max_gap_s * 1_000_000_000)

    for item in ordered:
        if not current:
            current = [item]
            continue
        prev_ts = int(current[-1]["timestamp"])
        this_ts = int(item["timestamp"])
        gap_too_large = this_ts - prev_ts > max_gap_ns
        chunk_full = len(current) >= max_frames
        if gap_too_large or chunk_full:
            chunks.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        chunks.append(current)

    segments = []
    dropped = []
    for idx, frames in enumerate(chunks):
        timestamps = [f["timestamp"] for f in frames]
        gaps = [
            (int(timestamps[i]) - int(timestamps[i - 1])) / 1_000_000_000
            for i in range(1, len(timestamps))
        ]
        segment = {
            "segment_id": f"seg_{idx:03d}",
            "frame_count": len(frames),
            "start_timestamp": timestamps[0],
            "end_timestamp": timestamps[-1],
            "max_gap_s": max(gaps) if gaps else 0.0,
            "frames": frames,
        }
        segments.append(segment)
        if len(frames) < min_frames:
            dropped.append(
                {
                    "segment_id": segment["segment_id"],
                    "reason": "frame_count_less_than_min_frames",
                    "frame_count": len(frames),
                    "timestamps": timestamps,
                }
            )

    return {
        "policy": {
            "use_pose_keyframes_only": True,
            "max_gap_s": max_gap_s,
            "min_frames": min_frames,
            "max_frames": max_frames,
        },
        "segments": segments,
        "dropped_segments": dropped,
    }


def build_intrinsics_only_camera_params(
    segment: dict[str, Any], calib: dict[str, Any]
) -> dict[str, Any]:
    intr = calib["intrinsics"]
    matrix = [
        [intr["fx"], 0.0, intr["cx"]],
        [0.0, intr["fy"], intr["cy"]],
        [0.0, 0.0, 1.0],
    ]
    frames = segment["frames"]
    return {
        "num_cameras": len(frames),
        "extrinsics": [],
        "intrinsics": [
            {"camera_id": str(frame["timestamp"]), "matrix": matrix}
            for frame in frames
        ],
    }


def renumber_valid_segments(
    split_report: dict[str, Any], min_frames: int
) -> list[dict[str, Any]]:
    valid_segments = []
    next_index = 0
    for segment in split_report["segments"]:
        if segment["frame_count"] < min_frames:
            continue
        renumbered = dict(segment)
        renumbered["source_segment_id"] = segment["segment_id"]
        renumbered["segment_id"] = f"seg_{next_index:03d}"
        valid_segments.append(renumbered)
        next_index += 1
    return valid_segments


def materialize_images(segment: dict[str, Any], out_image_dir: Path, mode: str) -> None:
    out_image_dir.mkdir(parents=True, exist_ok=True)
    for frame in segment["frames"]:
        src = Path(frame["image_path"])
        dst = out_image_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode == "symlink":
            dst.symlink_to(src)
        elif mode == "hardlink":
            os.link(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unsupported materialization mode: {mode}")


def build_qc_report(
    segment: dict[str, Any],
    segment_dir: Path,
    camera_params: dict[str, Any],
) -> dict[str, Any]:
    expected_stems = {str(frame["timestamp"]) for frame in segment["frames"]}
    image_paths = [
        segment_dir / "images" / Path(frame["image_path"]).name
        for frame in segment["frames"]
    ]
    intrinsics_ids = {str(item["camera_id"]) for item in camera_params.get("intrinsics", [])}
    extrinsics = camera_params.get("extrinsics", [])
    extrinsics_ids = {str(item["camera_id"]) for item in extrinsics}
    max_gap_s = segment.get("max_gap_s", 0.0)
    policy_max_gap_s = segment.get("policy", {}).get("max_gap_s")

    checks = {
        "image_files_exist": all(path.exists() for path in image_paths),
        "image_files_readable": all(path.exists() and path.stat().st_size > 0 for path in image_paths),
        "same_resolution": True,
        "timestamp_exact_match": all(
            Path(frame["image_path"]).stem == str(frame["timestamp"])
            for frame in segment["frames"]
        ),
        "camera_id_match_intrinsics": (
            camera_params.get("num_cameras") == len(expected_stems)
            and intrinsics_ids == expected_stems
        ),
        "camera_id_match_extrinsics": (
            None if not extrinsics else extrinsics_ids == expected_stems
        ),
        "max_gap_s_within_policy": (
            True if policy_max_gap_s is None else max_gap_s <= float(policy_max_gap_s)
        ),
        "pose_translation_metric": False,
    }
    required = [
        checks["image_files_exist"],
        checks["image_files_readable"],
        checks["same_resolution"],
        checks["timestamp_exact_match"],
        checks["camera_id_match_intrinsics"],
        checks["max_gap_s_within_policy"],
    ]
    passed = all(required)

    warnings = [
        "Extrinsics are disabled because pose.trans appears to be longitude/latitude/height."
    ]
    if not checks["image_files_exist"]:
        warnings.append("At least one materialized image is missing.")
    if not checks["camera_id_match_intrinsics"]:
        warnings.append("Intrinsics camera_id set does not exactly match image timestamp stems.")
    if not checks["max_gap_s_within_policy"]:
        warnings.append("Segment max timestamp gap exceeds the configured policy.")

    return {
        "passed": passed,
        "checks": checks,
        "warnings": warnings,
    }


def build_raw_audit(
    track_dir: Path,
    image_dir: Path,
    pose_path: Path,
    image_by_stem: dict[str, Path],
    poses: list[dict[str, Any]],
    calib: dict[str, Any],
) -> dict[str, Any]:
    stems = sorted(image_by_stem)
    pose_ts = [str(p.get("timestamp", p.get("name"))) for p in poses]
    return {
        "source_track_dir": str(track_dir),
        "image_dir": str(image_dir),
        "pose_path": str(pose_path),
        "camera_name": calib["camera_name"],
        "image_count": len(stems),
        "image_first_timestamp": stems[0],
        "image_last_timestamp": stems[-1],
        "pose_count": len(poses),
        "pose_first_timestamp": pose_ts[0] if pose_ts else None,
        "pose_last_timestamp": pose_ts[-1] if pose_ts else None,
        "image_resolution": [calib["image_width"], calib["image_height"]],
    }


def write_segment(
    segment: dict[str, Any],
    segment_dir: Path,
    calib: dict[str, Any],
    track_dir: Path,
    camera_name: str,
    pose_relpath: str,
    materialization: str,
) -> dict[str, Any]:
    materialize_images(segment, segment_dir / "images", materialization)
    save_json(segment_dir / "source_poses.json", [f["pose"] for f in segment["frames"]])
    save_text(
        segment_dir / "timestamps.txt",
        "\n".join(f["timestamp"] for f in segment["frames"]) + "\n",
    )

    camera_params = build_intrinsics_only_camera_params(segment, calib)
    intr_path = segment_dir / "camera_params_intrinsics_only.json"
    save_json(intr_path, camera_params)
    save_json(segment_dir / "camera_params.json", camera_params)

    meta = {
        "dataset_version": "mono_worldmirror_v0.2",
        "track_id": track_dir.name,
        "camera_name": camera_name,
        "segment_id": segment["segment_id"],
        "source_track_dir": str(track_dir),
        "image_source": f"camera/{camera_name}",
        "pose_source": pose_relpath,
        "frame_count": segment["frame_count"],
        "timestamp_start": segment["start_timestamp"],
        "timestamp_end": segment["end_timestamp"],
        "image_resolution": [calib["image_width"], calib["image_height"]],
        "image_materialization": materialization,
        "camera_prior": {
            "used_file": "camera_params.json",
            "mode": "intrinsics_only",
            "extrinsics_enabled": False,
        },
        "segmentation_policy": segment.get("policy"),
    }
    save_json(segment_dir / "meta.json", meta)

    qc_report = build_qc_report(segment, segment_dir, camera_params)
    save_json(segment_dir / "qc_report.json", qc_report)
    return meta


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    track_dir = args.track_dir.resolve()
    pose_path = track_dir / args.pose_relpath
    image_dir = track_dir / "camera" / args.camera_name
    dataset_root = args.output_root.resolve()
    track_out = dataset_root / "tracks" / track_dir.name / args.camera_name

    calib_data = load_json(track_dir / "calib.json")
    calib = extract_camera_calib(calib_data, args.camera_name)
    poses = load_json(pose_path)
    image_by_stem = load_image_index(image_dir)
    matches, match_report = match_poses_to_images(poses, image_dir)
    split_report = split_segments(
        matches,
        max_gap_s=args.max_gap_s,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )

    valid_segments = renumber_valid_segments(split_report, args.min_frames)
    raw_audit = build_raw_audit(track_dir, image_dir, pose_path, image_by_stem, poses, calib)

    if track_out.exists() and args.force:
        shutil.rmtree(track_out)
    track_out.mkdir(parents=True, exist_ok=True)

    save_json(track_out / "raw_audit.json", raw_audit)
    save_json(track_out / "source_calib.json", calib)
    save_json(track_out / "source_pose_all.json", poses)
    save_json(track_out / "timestamp_match_report.json", match_report)

    segments_public = []
    segment_metas = []
    for segment in split_report["segments"]:
        public_segment = {k: v for k, v in segment.items() if k != "frames"}
        public_segment["timestamps"] = [f["timestamp"] for f in segment["frames"]]
        segments_public.append(public_segment)

    for segment in valid_segments:
        segment["policy"] = split_report["policy"]
        meta = write_segment(
            segment,
            track_out / segment["segment_id"],
            calib,
            track_dir,
            args.camera_name,
            args.pose_relpath,
            args.materialization,
        )
        segment_metas.append(meta)

    save_json(
        track_out / "segments.json",
        {
            "policy": split_report["policy"],
            "segments": segments_public,
            "dropped_segments": split_report["dropped_segments"],
        },
    )

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_version": "mono_worldmirror_v0.2",
        "created_from": {
            "track_id": track_dir.name,
            "source_track_dir": str(track_dir),
            "camera_name": args.camera_name,
            "calib_file": "calib.json",
            "pose_file": args.pose_relpath,
        },
        "source_stats": {
            "raw_image_count": raw_audit["image_count"],
            "pose_count": raw_audit["pose_count"],
            "exact_match_count": match_report["exact_match_count"],
            "image_resolution": raw_audit["image_resolution"],
        },
        "generation_policy": {
            "use_pose_keyframes_only": True,
            "max_gap_s": args.max_gap_s,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "camera_prior_default": "intrinsics_only",
            "extrinsics_enabled": False,
            "image_materialization": args.materialization,
        },
        "tracks": [
            {
                "track_id": track_dir.name,
                "camera_name": args.camera_name,
                "relative_path": str(Path("tracks") / track_dir.name / args.camera_name),
                "segment_count": len(segment_metas),
            }
        ],
        "segments": [
            {
                "track_id": meta["track_id"],
                "camera_name": meta["camera_name"],
                "segment_id": meta["segment_id"],
                "relative_path": str(
                    Path("tracks") / meta["track_id"] / meta["camera_name"] / meta["segment_id"]
                ),
                "frame_count": meta["frame_count"],
                "camera_prior": meta["camera_prior"]["mode"],
            }
            for meta in segment_metas
        ],
    }
    save_json(dataset_root / "manifest.json", manifest)
    save_text(
        dataset_root / "README.md",
        "# HY-World Mono WorldMirror Dataset\n\n"
        "Generated by `scripts/make_mono_dataset.py`.\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track-dir",
        type=Path,
        default=Path(
            "/media/haiming/data2/2026_test/0521_ramp/7869148/pipeline/public/"
            "MX11-2S/HXMQBC7PZ4CT25SL1/2026-03-21/110247/110557/644470719"
        ),
    )
    parser.add_argument("--camera-name", default="mid_center_top_wide")
    parser.add_argument(
        "--pose-relpath",
        default="deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json",
    )
    parser.add_argument("--output-root", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset-name", default="hyworld2_mono_mid_center_top_wide_sample")
    parser.add_argument("--max-gap-s", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument(
        "--materialization",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    print(
        json.dumps(
            {
                "dataset_version": manifest["dataset_version"],
                "segment_count": len(manifest["segments"]),
                "manifest": str((args.output_root.resolve() / "manifest.json")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
