# HY-World 单目 WorldMirror 推理数据集结构

本文总结 `docs/data_format/dir_structure/mono_dataset_generation.md` 中定义并由 `scripts/make_mono_dataset.py` 生成的单目前馈式 3DGS 推理数据集结构。该数据集面向 HY-World 2.0 / WorldMirror 2.0 推理，不是训练集，也不是多相机融合数据集。

## 1. 根目录结构

推荐根目录命名为 `<mono_dataset_root>`，当前仓库默认生成到：

```text
dataset/
```

标准结构：

```text
<mono_dataset_root>/
├── manifest.json
├── README.md
└── tracks/
    └── <track_id>/
        └── <camera_name>/
            ├── raw_audit.json
            ├── source_calib.json
            ├── source_pose_all.json
            ├── timestamp_match_report.json
            ├── segments.json
            ├── seg_000/
            │   ├── images/
            │   ├── camera_params.json
            │   ├── camera_params_intrinsics_only.json
            │   ├── source_poses.json
            │   ├── timestamps.txt
            │   ├── meta.json
            │   └── qc_report.json
            ├── seg_001/
            └── ...
```

当前样例使用：

```text
track_id    = 644470719
camera_name = mid_center_top_wide
```

当前样例落盘路径：

```text
dataset/
└── tracks/
    └── 644470719/
        └── mid_center_top_wide/
            ├── seg_000/
            ├── seg_001/
            ├── seg_002/
            ├── seg_003/
            ├── seg_004/
            ├── seg_005/
            ├── seg_006/
            └── seg_007/
```

## 2. 顶层文件

| 文件 | 必须 | 说明 |
| --- | --- | --- |
| `manifest.json` | 是 | 数据集总索引，记录来源单程、相机、统计信息、制作策略和有效 segment 列表 |
| `README.md` | 推荐 | 数据集生成说明，便于独立拷贝数据集后快速识别来源 |
| `tracks/` | 是 | 按 `track_id/camera_name` 组织的实际数据 |

`manifest.json` 是批量推理、批量质检、可视化回溯的入口。下游不应直接扫描所有目录推断数据集，而应优先读取 `manifest.json`。

## 3. `manifest.json`

核心字段：

```json
{
  "dataset_name": "hyworld2_mono_mid_center_top_wide_sample",
  "dataset_version": "mono_worldmirror_v0.2",
  "created_from": {
    "track_id": "644470719",
    "source_track_dir": "/media/haiming/data2/2026_test/0521_ramp/7869148/pipeline/public/MX11-2S/HXMQBC7PZ4CT25SL1/2026-03-21/110247/110557/644470719",
    "camera_name": "mid_center_top_wide",
    "calib_file": "calib.json",
    "pose_file": "deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json"
  },
  "source_stats": {
    "raw_image_count": 5698,
    "pose_count": 191,
    "exact_match_count": 191,
    "image_resolution": [3840, 2160]
  },
  "generation_policy": {
    "use_pose_keyframes_only": true,
    "max_gap_s": 1.0,
    "min_frames": 2,
    "max_frames": 32,
    "camera_prior_default": "intrinsics_only",
    "extrinsics_enabled": false,
    "image_materialization": "symlink"
  },
  "segments": [
    {
      "track_id": "644470719",
      "camera_name": "mid_center_top_wide",
      "segment_id": "seg_000",
      "relative_path": "tracks/644470719/mid_center_top_wide/seg_000",
      "frame_count": 32,
      "camera_prior": "intrinsics_only"
    }
  ]
}
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `dataset_name` | 数据集名称 |
| `dataset_version` | 数据集结构版本，当前为 `mono_worldmirror_v0.2` |
| `created_from` | 原始单程、相机、标定和 SFM 位姿来源 |
| `source_stats` | 原始图像数量、位姿数量、exact match 数量和图像分辨率 |
| `generation_policy` | 制作策略，包括切段阈值、帧数上下限、prior 策略和图像落盘方式 |
| `tracks` | track/camera 级目录索引 |
| `segments` | 有效 segment 列表，下游推理应遍历该列表 |

当前样例 `segments` 共 8 个，有效帧数如下：

| segment | frame_count |
| --- | ---: |
| `seg_000` | 32 |
| `seg_001` | 9 |
| `seg_002` | 11 |
| `seg_003` | 18 |
| `seg_004` | 15 |
| `seg_005` | 32 |
| `seg_006` | 31 |
| `seg_007` | 32 |

## 4. Track/Camera 级文件

路径模板：

```text
tracks/<track_id>/<camera_name>/
```

当前样例：

```text
tracks/644470719/mid_center_top_wide/
```

文件职责：

| 文件 | 必须 | 说明 |
| --- | --- | --- |
| `raw_audit.json` | 是 | 原始输入审计结果，记录单程路径、图像目录、位姿路径、图像数量、时间范围、分辨率 |
| `source_calib.json` | 是 | 从原始 `calib.json` 中抽取出的目标相机标定 |
| `source_pose_all.json` | 是 | 原始 SFM 优化位姿副本或精简副本 |
| `timestamp_match_report.json` | 是 | 位姿和图像文件 stem 的匹配统计 |
| `segments.json` | 是 | 全量切段报告，包含有效段、被丢弃段和切段策略 |
| `seg_xxx/` | 是 | 可直接送入 WorldMirror 的一个推理片段 |

### 4.1 `raw_audit.json`

记录原始数据源和数量范围，例如：

```json
{
  "source_track_dir": ".../644470719",
  "image_dir": ".../camera/mid_center_top_wide",
  "pose_path": ".../deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json",
  "camera_name": "mid_center_top_wide",
  "image_count": 5698,
  "image_first_timestamp": "1774062167081105386",
  "image_last_timestamp": "1774062356981097387",
  "pose_count": 191,
  "pose_first_timestamp": "1774062167714436374",
  "pose_last_timestamp": "1774062356614437178",
  "image_resolution": [3840, 2160]
}
```

### 4.2 `source_calib.json`

记录目标相机标定。当前样例关键字段：

```json
{
  "camera_name": "mid_center_top_wide",
  "sensor_label": "SENSOR_LABEL_MID_CENTER_TOP_WIDE",
  "image_width": 3840,
  "image_height": 2160,
  "camera_model": "CAMERA_MODEL_KANNEL_BRAND",
  "intrinsics": {
    "fx": 1904.99609375,
    "fy": 1903.804931640625,
    "cx": 1917.16064453125,
    "cy": 1076.6253662109375
  },
  "distortion": {
    "k1": -0.029459979385137558,
    "k2": -0.005299502518028021,
    "k3": -0.0012424745364114642,
    "k4": 0.0012543649645522237
  }
}
```

注意：当前相机是 KB/Kannala-Brandt 广角模型。WorldMirror 的 `prior_cam_path` 只表达 pinhole 3x3 K，因此正式高精度几何使用前建议先做去畸变并写入新 K。

### 4.3 `timestamp_match_report.json`

记录图像和位姿匹配结果。当前样例核心结论：

```json
{
  "image_count": 5698,
  "pose_count": 191,
  "exact_match_count": 191,
  "missing_pose_timestamps": [],
  "nearest_error_ns": {
    "min": 0,
    "max": 0,
    "mean": 0
  },
  "first_matched_image_index": 19,
  "last_matched_image_index": 5686
}
```

### 4.4 `segments.json`

记录切段策略和所有原始切段。有效 segment 在输出目录中会重新连续编号为 `seg_000...seg_007`；被丢弃的短段保留在 `dropped_segments` 中，便于追溯。

核心字段：

```json
{
  "policy": {
    "use_pose_keyframes_only": true,
    "max_gap_s": 1.0,
    "min_frames": 2,
    "max_frames": 32
  },
  "segments": [],
  "dropped_segments": []
}
```

## 5. Segment 级结构

路径模板：

```text
tracks/<track_id>/<camera_name>/seg_xxx/
```

标准结构：

```text
seg_xxx/
├── images/
│   ├── <timestamp_ns>.jpeg
│   └── ...
├── camera_params.json
├── camera_params_intrinsics_only.json
├── source_poses.json
├── timestamps.txt
├── meta.json
└── qc_report.json
```

文件职责：

| 文件 | 必须 | 说明 |
| --- | --- | --- |
| `images/` | 是 | WorldMirror 输入图像目录，只包含当前 segment 的图像 |
| `camera_params.json` | 推荐 | WorldMirror 实际使用的 camera prior 文件 |
| `camera_params_intrinsics_only.json` | 推荐 | 只包含内参的安全先验 |
| `source_poses.json` | 是 | 当前 segment 对应的原始 SFM 位姿 |
| `timestamps.txt` | 是 | 当前 segment 的图片时间戳，一行一个 stem |
| `meta.json` | 是 | 当前 segment 的来源、帧数、时间范围和 prior 策略 |
| `qc_report.json` | 是 | 当前 segment 的自动质检结果 |

当前样例默认使用软链接方式落盘图片：

```text
image_materialization = symlink
```

因此 `images/` 中的 JPEG 文件通常指向原始单程目录，不复制原始大图。

## 6. Segment 文件格式

### 6.1 `images/`

命名规则：

```text
images/<timestamp_ns>.jpeg
```

要求：

- 文件 stem 必须是纳秒时间戳。
- 文件 stem 应与 `source_poses.json` 中的 `timestamp` 或 `name` 一致。
- 同一 segment 内图像宽高和宽高比应一致。
- 若使用 `camera_params.json`，每张图都必须有对应的 `camera_id`。

### 6.2 `camera_params.json`

当前样例的 `camera_params.json` 与 `camera_params_intrinsics_only.json` 内容相同，只注入内参，不注入外参：

```json
{
  "num_cameras": 2,
  "extrinsics": [],
  "intrinsics": [
    {
      "camera_id": "1774062167714436374",
      "matrix": [
        [1904.99609375, 0.0, 1917.16064453125],
        [0.0, 1903.804931640625, 1076.6253662109375],
        [0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

关键约束：

- `num_cameras` 必须等于 `images/` 中图片数量。
- `intrinsics` 的 `camera_id` 集合必须与图片 stem 集合完全一致。
- `extrinsics` 当前保持为空，因为样例 `pose.trans` 疑似经纬高，不能直接作为米制 c2w 平移。
- 如果未来启用外参，`extrinsics.matrix` 必须是 OpenCV camera convention 下的 4x4 camera-to-world。

### 6.3 `source_poses.json`

保存当前 segment 的原始 SFM 位姿记录。单条记录示例：

```json
{
  "idx": 0,
  "name": "1774062167714436374",
  "pose": {
    "quat": [-0.3630918492142215, -0.006212254122689151, -0.0054377466595192945, 0.9317167744782634],
    "trans": [120.14875771995887, 30.305922915097817, 11.83336337501263]
  },
  "timestamp": 1774062167714436374,
  "submap_id": 0,
  "track_id": -1,
  "version": "2.4.0",
  "optimized_source": 1
}
```

注意：`pose.trans` 当前不应直接注入到 `camera_params.json`。它的数值更像 `longitude / latitude / height`。

### 6.4 `timestamps.txt`

纯文本文件，一行一个图像时间戳：

```text
1774062167714436374
1774062167947755496
```

用途：

- 快速检查 segment 帧序。
- 与 `images/` 文件名、`source_poses.json` 和 `camera_params.json` 对齐。

### 6.5 `meta.json`

记录 segment 元数据。核心字段：

```json
{
  "dataset_version": "mono_worldmirror_v0.2",
  "track_id": "644470719",
  "camera_name": "mid_center_top_wide",
  "segment_id": "seg_001",
  "source_segment_id": "seg_001",
  "source_track_dir": ".../644470719",
  "image_source": "camera/mid_center_top_wide",
  "pose_source": "deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json",
  "frame_count": 9,
  "timestamp_start": "1774062175114422955",
  "timestamp_end": "1774062178281101073",
  "image_resolution": [3840, 2160],
  "image_materialization": "symlink",
  "camera_prior": {
    "used_file": "camera_params.json",
    "mode": "intrinsics_only",
    "extrinsics_enabled": false
  },
  "segmentation_policy": {
    "use_pose_keyframes_only": true,
    "max_gap_s": 1.0,
    "min_frames": 2,
    "max_frames": 32
  }
}
```

`source_segment_id` 表示原始切段编号。有效 segment 会重新连续编号，因此 `segment_id` 是对外稳定编号，`source_segment_id` 用于追溯原始切段。

### 6.6 `qc_report.json`

记录自动质检结果：

```json
{
  "passed": true,
  "checks": {
    "image_files_exist": true,
    "image_files_readable": true,
    "same_resolution": true,
    "timestamp_exact_match": true,
    "camera_id_match_intrinsics": true,
    "camera_id_match_extrinsics": null,
    "max_gap_s_within_policy": true,
    "pose_translation_metric": false
  },
  "warnings": [
    "Extrinsics are disabled because pose.trans appears to be longitude/latitude/height."
  ]
}
```

其中：

- `camera_id_match_intrinsics=true` 表示 `camera_params.json` 的内参覆盖所有输入图像。
- `camera_id_match_extrinsics=null` 表示当前没有提供外参。
- `pose_translation_metric=false` 是当前样例的关键风险提示：SFM 平移疑似经纬高，不应直接作为米制平移使用。

## 7. 与 WorldMirror 推理的关系

单个 segment 可直接作为 WorldMirror 输入：

```bash
conda run -n hyworld2 bash -lc '
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=0 python -m hyworld2.worldrecon.pipeline \
  --input_path dataset/tracks/644470719/mid_center_top_wide/seg_001/images \
  --prior_cam_path dataset/tracks/644470719/mid_center_top_wide/seg_001/camera_params.json \
  --strict_output_path outputs/worldmirror_mono_seg001_smoke \
  --target_size 392 \
  --enable_bf16 \
  --no_sky_mask \
  --no_edge_mask \
  --no_interactive
'
```

本数据集结构中：

- `images/` 对应 WorldMirror 的 `--input_path`。
- `camera_params.json` 对应 WorldMirror 的 `--prior_cam_path`。
- `manifest.json` 可用于批量遍历多个 segment。
- `meta.json` 和 `qc_report.json` 用于推理前过滤不合格 segment。

## 8. 使用注意

1. 当前数据集只默认注入 intrinsics，不注入 extrinsics。
2. 当前相机是 KB 广角模型，`camera_params.json` 中的 K 是 pinhole 近似；高精度版本建议先去畸变。
3. `dataset/` 和 `outputs/` 已被 `.gitignore` 忽略，数据集产物默认不进入 Git。
4. 批量推理应读取 `manifest.json` 的 `segments` 列表，而不是直接扫描目录。
5. 如果复制数据集到其它机器，软链接图片可能失效；需要使用 copy 模式重新生成或同步原始单程目录。
