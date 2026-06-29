#!/usr/bin/env python3
"""Build mono WorldMirror segments and generate Mask2Former-derived masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_mono_dataset import build_dataset

IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".webp")
DEFAULT_MODEL_PATH = Path(
    "/home/haiming/admap_ws/seg_mask/mask2former/weights/"
    "mask2former_r50_cadata_1920x1080_90000.pt"
)

CLASS_LABELS = [
    "road",
    "sidewalk",
    "vegetation",
    "terrain",
    "pole",
    "traffic_sign",
    "traffic_light",
    "sign_line",
    "lane_marking",
    "person",
    "rider",
    "bicycle",
    "tricycle",
    "car",
    "building",
    "fence",
    "sky",
    "traffic_cone",
    "bollard",
    "guide_post",
    "crosswalk_line",
    "traffic_arrow",
    "guide_line",
    "stop_line",
    "slowdown_triangle",
    "speed_sign",
    "diamond",
    "bicycle_sign",
    "speed_bumps",
    "traversable_obstruction",
    "untraversable_obstruction",
    "ego_mask",
    "other",
    "parking_column",
    "no_parking_line",
    "slow_down_line",
    "road_text",
    "stop_attention_line",
    "background",
]

SKY_CLASS_ID = 16
EGO_CLASS_ID = 31


class Segmenter(Protocol):
    def infer(self, images_bgr: Sequence[np.ndarray]) -> list[np.ndarray]:
        ...


@dataclass(frozen=True)
class SegmentFrame:
    stem: str
    image_path: Path
    intrinsic: np.ndarray


@dataclass(frozen=True)
class SegmentInferenceResult:
    segment_id: str
    planned_frames: int
    written_semantic: int
    written_masks: int
    dry_run: bool


class Mask2FormerSegmenter:
    """TorchScript Mask2Former wrapper compatible with the seg_mask V1 model."""

    def __init__(self, model_path: Path, input_wh: tuple[int, int], device: str = "auto"):
        import torch
        import torch.nn.functional as functional

        if not model_path.exists():
            raise FileNotFoundError(f"Mask2Former model not found: {model_path}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.functional = functional
        self.device = torch.device(device)
        self.input_wh = input_wh
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        self.mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        self.std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)

    def infer(self, images_bgr: Sequence[np.ndarray]) -> list[np.ndarray]:
        tensor, metas = self._encode(images_bgr)
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            cls_pred_list, mask_pred_list = self.model(tensor)
            mask_cls_results = cls_pred_list[-1]
            mask_pred_results = mask_pred_list[-1]
            mask_pred_results = self.functional.interpolate(
                mask_pred_results,
                size=(self.input_wh[1], self.input_wh[0]),
                mode="bilinear",
                align_corners=False,
            )
            cls_score = self.functional.softmax(mask_cls_results, dim=-1)[..., :-1]
            mask_pred = mask_pred_results.sigmoid()
            seg_logits = self.torch.einsum("bqc,bqhw->bchw", cls_score, mask_pred)
            pred = seg_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        return self._decode(pred, metas)

    def _encode(self, images_bgr: Sequence[np.ndarray]):
        resized = []
        metas = []
        for image in images_bgr:
            original_wh = (image.shape[1], image.shape[0])
            if original_wh != self.input_wh:
                scale = max(original_wh[0] / self.input_wh[0], original_wh[1] / self.input_wh[1])
                new_wh = (int(original_wh[0] / scale), int(original_wh[1] / scale))
                left = (self.input_wh[0] - new_wh[0]) // 2
                top = (self.input_wh[1] - new_wh[1]) // 2
                canvas = np.zeros((self.input_wh[1], self.input_wh[0], 3), dtype=np.uint8)
                canvas[top : top + new_wh[1], left : left + new_wh[0]] = cv2.resize(image, new_wh)
                roi_ltrb = (left, top, left + new_wh[0], top + new_wh[1])
            else:
                canvas = image
                roi_ltrb = (0, 0, original_wh[0], original_wh[1])
            metas.append({"original_wh": original_wh, "roi_ltrb": roi_ltrb})
            resized.append(canvas[..., ::-1])

        tensor = self.torch.tensor(np.asarray(resized), dtype=self.torch.float32)
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor, metas

    @staticmethod
    def _decode(predictions: np.ndarray, metas: Sequence[dict[str, Any]]) -> list[np.ndarray]:
        masks = []
        for idx, pred in enumerate(predictions):
            left, top, right, bottom = metas[idx]["roi_ltrb"]
            original_wh = metas[idx]["original_wh"]
            pred = pred[top:bottom, left:right]
            masks.append(cv2.resize(pred, original_wh, interpolation=cv2.INTER_NEAREST))
        return masks


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def save_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def default_model_path() -> Path:
    return Path(os.environ.get("MASK2FORMER_MODEL_PATH", DEFAULT_MODEL_PATH))


def read_timestamps(segment_dir: Path, limit: int | None = None) -> list[str]:
    path = segment_dir / "timestamps.txt"
    if not path.exists():
        raise FileNotFoundError(f"timestamps.txt not found: {path}")
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return stems[:limit] if limit is not None else stems


def index_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    images = {}
    for path in image_dir.iterdir():
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            images[path.stem] = path
    return images


def load_distortion(source_calib_path: Path) -> np.ndarray:
    calib = load_json(source_calib_path)
    dist = calib.get("distortion", {})
    return np.array(
        [
            float(dist.get("k1", 0.0)),
            float(dist.get("k2", 0.0)),
            float(dist.get("k3", 0.0)),
            float(dist.get("k4", 0.0)),
        ],
        dtype=np.float64,
    )


def collect_segment_frames(segment_dir: Path, limit: int | None = None) -> list[SegmentFrame]:
    stems = read_timestamps(segment_dir, limit=limit)
    images_by_stem = index_images(segment_dir / "images")
    camera_params = load_json(segment_dir / "camera_params.json")
    intrinsics = {
        str(item["camera_id"]): np.asarray(item["matrix"], dtype=np.float64)
        for item in camera_params.get("intrinsics", [])
    }

    frames: list[SegmentFrame] = []
    missing_images = []
    missing_intrinsics = []
    for stem in stems:
        image_path = images_by_stem.get(stem)
        intrinsic = intrinsics.get(stem)
        if image_path is None:
            missing_images.append(stem)
            continue
        if intrinsic is None:
            missing_intrinsics.append(stem)
            continue
        frames.append(SegmentFrame(stem=stem, image_path=image_path, intrinsic=intrinsic))

    if missing_images:
        raise FileNotFoundError(f"missing segment images for timestamps: {missing_images[:5]}")
    if missing_intrinsics:
        raise ValueError(f"missing camera intrinsics for timestamps: {missing_intrinsics[:5]}")
    if not frames:
        raise RuntimeError(f"no frames collected from {segment_dir}")
    return frames


def undistort_image(image: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        intrinsic,
        distortion,
        np.eye(3),
        intrinsic,
        (width, height),
        cv2.CV_16SC2,
    )
    return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [[(idx * 37) % 256, (idx * 67) % 256, (idx * 97) % 256] for idx in range(256)],
        dtype=np.uint8,
    )
    return palette[mask]


def mask_stats(stem: str, mask: np.ndarray) -> dict[str, Any]:
    labels, counts = np.unique(mask, return_counts=True)
    return {
        "stem": stem,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "labels": {str(int(label)): int(count) for label, count in zip(labels, counts)},
    }


def semantic_boundary(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros(mask.shape, dtype=bool)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[1:, :] != mask[:-1, :]
    return edge


def dilate_binary(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def derive_masks_from_semantic(semantic: np.ndarray, edge_dilation: int = 3) -> dict[str, np.ndarray]:
    sky = semantic == SKY_CLASS_ID
    ego = semantic == EGO_CLASS_ID
    edge = semantic_boundary(semantic)
    edge = dilate_binary(edge, edge_dilation)
    keep = (~sky) & (~ego) & (~edge)
    return {
        "sky_mask": sky.astype(np.uint8) * 255,
        "ego_mask": ego.astype(np.uint8) * 255,
        "semantic_edge_mask": edge.astype(np.uint8) * 255,
        "worldmirror_keep_mask": keep.astype(np.uint8) * 255,
    }


def write_binary_masks(
    segment_dir: Path,
    stem: str,
    semantic: np.ndarray,
    edge_dilation: int,
    overwrite: bool,
) -> bool:
    outputs = derive_masks_from_semantic(semantic, edge_dilation=edge_dilation)
    wrote = False
    for name, mask in outputs.items():
        out_path = segment_dir / "masks" / name / f"{stem}.png"
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), mask)
        wrote = True
    return wrote


def build_image_manifest(
    segment_dir: Path,
    source_calib_path: Path,
    model_path: Path,
    frames: Sequence[SegmentFrame],
    input_wh: tuple[int, int],
    undistort: bool,
    save_color_viz: bool,
    written_semantic: int,
    dry_run: bool,
) -> dict[str, Any]:
    frame_rows = []
    for frame in frames:
        row = {
            "timestamp_ns": frame.stem,
            "image": str(Path("images") / frame.image_path.name),
            "semantic": str(Path("image_segment") / "semantic" / f"{frame.stem}.png"),
            "undistorted": str(Path("image_segment") / "undistorted" / f"{frame.stem}.jpeg"),
        }
        if save_color_viz:
            row["color_viz"] = str(Path("image_segment") / "color_viz" / f"{frame.stem}.png")
        frame_rows.append(row)

    return {
        "pipeline": "mask2former_segment_v1",
        "segment_dir": segment_dir.name,
        "camera_name": load_json(segment_dir / "meta.json").get("camera_name") if (segment_dir / "meta.json").exists() else None,
        "input_images": "images",
        "camera_params": "camera_params.json",
        "calibration_source": relpath(source_calib_path, segment_dir),
        "undistort": undistort,
        "model_path": str(model_path),
        "input_wh": [input_wh[0], input_wh[1]],
        "class_labels": "class_labels.json",
        "dry_run": dry_run,
        "counts": {
            "planned_frames": len(frames),
            "written_semantic": written_semantic,
        },
        "outputs": {
            "semantic_dir": "image_segment/semantic",
            "undistorted_dir": "image_segment/undistorted",
            "color_viz_dir": "image_segment/color_viz" if save_color_viz else None,
        },
        "frames": frame_rows,
    }


def build_masks_manifest(edge_dilation: int, written_masks: int, dry_run: bool) -> dict[str, Any]:
    return {
        "pipeline": "semantic_masks_from_mask2former_v1",
        "semantic_source": "image_segment/semantic",
        "class_ids": {
            "sky": SKY_CLASS_ID,
            "ego_mask": EGO_CLASS_ID,
        },
        "edge_dilation": edge_dilation,
        "dry_run": dry_run,
        "counts": {
            "written_masks": written_masks,
        },
        "pixel_conventions": {
            "sky_mask": "255=sky, 0=non-sky",
            "ego_mask": "255=ego/self vehicle, 0=other",
            "semantic_edge_mask": "255=edge/filter candidate, 0=non-edge",
            "worldmirror_keep_mask": "255=keep, 0=filter out",
        },
        "outputs": {
            "sky_mask_dir": "masks/sky_mask",
            "ego_mask_dir": "masks/ego_mask",
            "semantic_edge_mask_dir": "masks/semantic_edge_mask",
            "worldmirror_keep_mask_dir": "masks/worldmirror_keep_mask",
        },
    }


def iter_batches(items: Sequence[SegmentFrame], batch_size: int) -> Iterable[Sequence[SegmentFrame]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_segment_pipeline(
    segment_dir: Path,
    source_calib_path: Path | None = None,
    model_path: Path | None = None,
    segmenter: Segmenter | None = None,
    batch_size: int = 1,
    input_wh: tuple[int, int] = (1920, 1080),
    device: str = "auto",
    limit: int | None = None,
    edge_dilation: int = 3,
    undistort: bool = True,
    save_color_viz: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> SegmentInferenceResult:
    segment_dir = segment_dir.resolve()
    source_calib_path = (source_calib_path or segment_dir.parent / "source_calib.json").resolve()
    model_path = (model_path or default_model_path()).resolve()
    frames = collect_segment_frames(segment_dir, limit=limit)

    for subdir in (
        segment_dir / "image_segment" / "semantic",
        segment_dir / "image_segment" / "undistorted",
        segment_dir / "masks" / "sky_mask",
        segment_dir / "masks" / "ego_mask",
        segment_dir / "masks" / "semantic_edge_mask",
        segment_dir / "masks" / "worldmirror_keep_mask",
    ):
        subdir.mkdir(parents=True, exist_ok=True)
    if save_color_viz:
        (segment_dir / "image_segment" / "color_viz").mkdir(parents=True, exist_ok=True)

    save_json(
        segment_dir / "image_segment" / "class_labels.json",
        {str(idx): name for idx, name in enumerate(CLASS_LABELS)},
    )

    if dry_run:
        save_json(
            segment_dir / "image_segment" / "manifest.json",
            build_image_manifest(
                segment_dir,
                source_calib_path,
                model_path,
                frames,
                input_wh,
                undistort,
                save_color_viz,
                written_semantic=0,
                dry_run=True,
            ),
        )
        save_json(segment_dir / "masks" / "manifest.json", build_masks_manifest(edge_dilation, 0, True))
        return SegmentInferenceResult(segment_dir.name, len(frames), 0, 0, True)

    if undistort and not source_calib_path.exists():
        raise FileNotFoundError(f"source calibration not found: {source_calib_path}")
    distortion = load_distortion(source_calib_path) if undistort else np.zeros(4, dtype=np.float64)
    if segmenter is None:
        segmenter = Mask2FormerSegmenter(model_path, input_wh=input_wh, device=device)

    stats = []
    written_semantic = 0
    written_masks = 0

    for batch in iter_batches(frames, batch_size):
        infer_frames = []
        images = []
        for frame in batch:
            semantic_path = segment_dir / "image_segment" / "semantic" / f"{frame.stem}.png"
            if semantic_path.exists() and not overwrite:
                semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
                if semantic is None:
                    raise RuntimeError(f"failed to read existing semantic mask: {semantic_path}")
                if write_binary_masks(segment_dir, frame.stem, semantic, edge_dilation, overwrite=False):
                    written_masks += 1
                stats.append(mask_stats(frame.stem, semantic))
                continue

            image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read image: {frame.image_path}")
            if undistort:
                image = undistort_image(image, frame.intrinsic, distortion)
            undistorted_path = segment_dir / "image_segment" / "undistorted" / f"{frame.stem}.jpeg"
            cv2.imwrite(str(undistorted_path), image)
            infer_frames.append(frame)
            images.append(image)

        if not infer_frames:
            continue

        semantic_masks = segmenter.infer(images)
        for frame, semantic in zip(infer_frames, semantic_masks):
            semantic_path = segment_dir / "image_segment" / "semantic" / f"{frame.stem}.png"
            cv2.imwrite(str(semantic_path), semantic)
            written_semantic += 1
            if save_color_viz:
                color_path = segment_dir / "image_segment" / "color_viz" / f"{frame.stem}.png"
                cv2.imwrite(str(color_path), colorize_mask(semantic))
            if write_binary_masks(segment_dir, frame.stem, semantic, edge_dilation, overwrite=True):
                written_masks += 1
            stats.append(mask_stats(frame.stem, semantic))

    save_jsonl(segment_dir / "image_segment" / "mask_stats.jsonl", stats)
    save_json(
        segment_dir / "image_segment" / "manifest.json",
        build_image_manifest(
            segment_dir,
            source_calib_path,
            model_path,
            frames,
            input_wh,
            undistort,
            save_color_viz,
            written_semantic=written_semantic,
            dry_run=False,
        ),
    )
    save_json(
        segment_dir / "masks" / "manifest.json",
        build_masks_manifest(edge_dilation=edge_dilation, written_masks=written_masks, dry_run=False),
    )

    return SegmentInferenceResult(
        segment_id=segment_dir.name,
        planned_frames=len(frames),
        written_semantic=written_semantic,
        written_masks=written_masks,
        dry_run=False,
    )


def discover_segments(track_dataset_dir: Path, segment_ids: Sequence[str] | None = None) -> list[Path]:
    if segment_ids:
        segments = [track_dataset_dir / segment_id for segment_id in segment_ids]
    else:
        segments = sorted(path for path in track_dataset_dir.glob("seg_*") if path.is_dir())
    missing = [str(path) for path in segments if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"segment directories not found: {missing}")
    if not segments:
        raise FileNotFoundError(f"no segment directories found under {track_dataset_dir}")
    return segments


def build_dataset_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        track_dir=args.track_dir,
        camera_name=args.camera_name,
        pose_relpath=args.pose_relpath,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        max_gap_s=args.max_gap_s,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        materialization=args.materialization,
        force=args.force,
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if not args.skip_build_dataset:
        dataset_manifest = build_dataset(build_dataset_namespace(args))
    else:
        dataset_manifest = None

    if args.dataset_track_dir is not None:
        track_dataset_dir = args.dataset_track_dir.resolve()
    else:
        track_dataset_dir = (
            args.output_root.resolve()
            / "tracks"
            / args.track_dir.resolve().name
            / args.camera_name
        )

    source_calib_path = (args.source_calib or track_dataset_dir / "source_calib.json").resolve()
    segment_dirs = discover_segments(track_dataset_dir, args.segment_id)
    if args.limit_segments is not None:
        segment_dirs = segment_dirs[: args.limit_segments]

    model_path = (args.model_path or default_model_path()).resolve()
    segmenter = None
    if not args.dry_run:
        segmenter = Mask2FormerSegmenter(
            model_path,
            input_wh=(args.input_width, args.input_height),
            device=args.device,
        )

    results = []
    for segment_dir in segment_dirs:
        result = run_segment_pipeline(
            segment_dir=segment_dir,
            source_calib_path=source_calib_path,
            model_path=model_path,
            segmenter=segmenter,
            batch_size=args.batch_size,
            input_wh=(args.input_width, args.input_height),
            device=args.device,
            limit=args.limit_frames,
            edge_dilation=args.edge_dilation,
            undistort=not args.no_undistort,
            save_color_viz=args.save_color_viz,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        results.append(result)

    pipeline_manifest = {
        "pipeline": "mono_dataset_mask_generation_v1",
        "dataset_manifest": dataset_manifest,
        "track_dataset_dir": str(track_dataset_dir),
        "source_calib": str(source_calib_path),
        "model_path": str(model_path),
        "dry_run": args.dry_run,
        "segment_count": len(results),
        "segments": [
            {
                "segment_id": item.segment_id,
                "planned_frames": item.planned_frames,
                "written_semantic": item.written_semantic,
                "written_masks": item.written_masks,
                "dry_run": item.dry_run,
            }
            for item in results
        ],
    }
    save_json(track_dataset_dir / "mask_generation_manifest.json", pipeline_manifest)
    return pipeline_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--skip-build-dataset", action="store_true")
    parser.add_argument("--dataset-track-dir", type=Path, default=None)

    parser.add_argument("--source-calib", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--segment-id", action="append", default=None)
    parser.add_argument("--limit-segments", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-width", type=int, default=1920)
    parser.add_argument("--input-height", type=int, default=1080)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--edge-dilation", type=int, default=3)
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--save-color-viz", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run_pipeline(parse_args(argv))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
