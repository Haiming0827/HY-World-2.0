# HY-World 单目前馈式 3DGS 推理数据集制作流程

版本：V0.2

本文说明如何从单程数据目录制作 HY-World 2.0 / WorldMirror 2.0 可用的单目前馈式 3DGS 推理数据集。本文重点是推理数据集，不是训练集，也不是多相机融合数据集。

样例单程目录：

```text
/media/haiming/data2/2026_test/0521_ramp/7869148/pipeline/public/MX11-2S/HXMQBC7PZ4CT25SL1/2026-03-21/110247/110557/644470719
```

目标相机：

```text
mid_center_top_wide
```

目标 SFM 优化位姿：

```text
deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json
```

核心结论先写在前面：

- 图像和 SFM 关键帧时间戳可以对上；当前样例为 `191/191` exact match。
- 正式制作时建议先抽取 SFM 关键帧图像，不要把全量 30 FPS 图像一次性送入 WorldMirror。
- 关键帧存在长时间间隔，需要按时间连续性切成多个 segment。
- 当前样例的 `pose.trans` 数值疑似 `longitude / latitude / height`，不能直接作为 WorldMirror `camera_params.json` 的米制 c2w 平移。
- 第一版可交付数据集建议只生成 intrinsics-only camera prior，或完全不注入 camera prior；外参注入必须在位姿语义、坐标单位、坐标轴约定确认之后再开启。

## 1. 制作目标

制作结果应满足三个层次：

| 层次 | 目标 | 是否必须 |
| --- | --- | --- |
| 原始审计层 | 保存单程输入、标定、位姿、时间戳匹配、切段依据，保证可追溯 | 必须 |
| WorldMirror 输入层 | 生成每个 segment 的 `images/` 和可选 `camera_params.json`，可直接作为 `--input_path` | 必须 |
| 推理结果层 | 保存 WorldMirror 输出的 `camera_params.json/depth/normal/points.ply/gaussians.ply` 等 | 推荐 |
| 可视化和质检层 | 保存 Rerun/Open3D/SuperSplat/CloudCompare 可读的质检说明和截图索引 | 推荐 |

最终数据集不应只是一堆图片。它至少要回答：

1. 图片来自哪个单程、哪个相机、哪个时间段。
2. 这些图片是否都有对应位姿。
3. 位姿是否被注入到 WorldMirror，若没有注入，原因是什么。
4. 图像是否做过去畸变、裁剪、缩放或重命名。
5. 推理时使用了哪些 WorldMirror 参数。
6. 输出是否完整，`gaussians.ply` 和 `points.ply` 能否打开。

## 2. 推荐目录结构

建议把“原始审计信息”和“每个推理 segment”分开组织：

```text
<mono_dataset_root>/
├── manifest.json
├── README.md
└── tracks/
    └── 644470719/
        └── mid_center_top_wide/
            ├── raw_audit.json
            ├── source_calib.json
            ├── source_pose_all.json
            ├── timestamp_match_report.json
            ├── segments.json
            ├── seg_000/
            │   ├── images/
            │   │   ├── 1774062167714436374.jpeg
            │   │   └── 1774062167947755496.jpeg
            │   ├── camera_params.json
            │   ├── camera_params_intrinsics_only.json
            │   ├── camera_params_full.json
            │   ├── source_poses.json
            │   ├── timestamps.txt
            │   ├── meta.json
            │   └── qc_report.json
            ├── seg_001/
            └── seg_002/
```

文件职责：

| 文件 | 职责 |
| --- | --- |
| `manifest.json` | 数据集总索引，记录版本、来源、相机、segment 列表、制作参数 |
| `raw_audit.json` | 原始单程级审计结果，含输入路径、文件数量、时间范围、分辨率 |
| `source_calib.json` | 从 `calib.json` 抽取出的目标相机标定 |
| `source_pose_all.json` | 复制或精简保存原始 SFM 优化位姿，方便脱离原始目录复查 |
| `timestamp_match_report.json` | 图像与位姿匹配统计、缺失项、最近邻误差、长间隔统计 |
| `segments.json` | 所有 segment 的切分结果和切分原因 |
| `seg_xxx/images/` | WorldMirror 实际输入图像目录 |
| `seg_xxx/camera_params.json` | WorldMirror 实际使用的 camera prior；可以软链或复制自 intrinsics-only/full 版本 |
| `seg_xxx/camera_params_intrinsics_only.json` | 只包含内参的安全先验 |
| `seg_xxx/camera_params_full.json` | 包含内外参的候选先验；仅在坐标系确认后生成和使用 |
| `seg_xxx/source_poses.json` | 当前 segment 对应的 SFM 位姿原始记录 |
| `seg_xxx/timestamps.txt` | 当前 segment 的图像时间戳，一行一个 stem |
| `seg_xxx/meta.json` | 当前 segment 的制作配置和来源信息 |
| `seg_xxx/qc_report.json` | 当前 segment 的自动质检结果 |

`camera_params_full.json` 在当前样例中应作为“候选文件”而不是默认输入，因为 `pose.trans` 尚不能确认为米制世界坐标。

## 3. 当前样例已验证事实

当前样例的关键统计：

| 项 | 值 |
| --- | ---: |
| 图像目录 | `camera/mid_center_top_wide` |
| 图像数量 | 5698 |
| 图像格式 | RGB JPEG |
| 图像分辨率 | 3840x2160 |
| 图像首帧时间戳 | `1774062167081105386` |
| 图像末帧时间戳 | `1774062356981097387` |
| 图像时间间隔中位数 | `33.334339 ms` |
| SFM 优化位姿数量 | 191 |
| SFM 首个位姿时间戳 | `1774062167714436374` |
| SFM 最后位姿时间戳 | `1774062356614437178` |
| 位姿与图像 exact match | `191/191` |
| 位姿最近邻误差 | 全部 `0 ns` |
| 位姿图像索引范围 | 第 19 张到第 5686 张 |
| 首个位姿相对图像首帧偏移 | `633.330988 ms` |
| 末帧图像晚于最后位姿 | `366.660209 ms` |

已发现的大间隔包括：

| 位姿跳变 | 间隔 | 图像步长 |
| --- | ---: | ---: |
| 40 -> 41 | `1.566670347 s` | 47 |
| 41 -> 42 | `12.666656965 s` | 380 |
| 42 -> 43 | `1.166664497 s` | 35 |
| 53 -> 54 | `5.233349158 s` | 157 |
| 54 -> 55 | `15.933325581 s` | 478 |
| 72 -> 73 | `7.266649663 s` | 218 |
| 73 -> 74 | `1.200007292 s` | 36 |
| 88 -> 89 | `1.199999089 s` | 36 |
| 89 -> 90 | `1.266669344 s` | 38 |
| 90 -> 91 | `20.333327470 s` | 610 |
| 91 -> 92 | `1.066680945 s` | 32 |
| 92 -> 93 | `1.099996715 s` | 33 |
| 155 -> 156 | `1.166656194 s` | 35 |
| 156 -> 157 | `1.699987714 s` | 51 |
| 157 -> 158 | `44.700002705 s` | 1341 |

`pose.trans` 数值范围：

```text
min = [120.14875771995887, 30.304571737376648, 11.829010959798545]
max = [120.15011920034881, 30.305922915097817, 17.80796624721914]
```

它非常接近经纬高，而不是米制局部 XYZ。根目录 `loc.json` 首条记录中也存在接近的 `longitude/latitude/height`。因此不要直接把 `pose.trans` 写入 WorldMirror 的 c2w 平移。

## 4. WorldMirror 输入约束

根据仓库中的 `hyworld2/worldrecon/hyworldmirror/utils/inference_utils.py` 和 `DOCUMENTATION.md`，制作数据集时必须遵守以下约束。

### 4.1 图像输入约束

WorldMirror 从目录读取图片时，会按以下后缀收集并排序：

```text
*.jpeg
*.jpg
*.png
*.webp
```

制作要求：

- 每个 segment 的 `images/` 只放本次推理需要的图片。
- 图片文件名建议保留原始时间戳 stem，例如 `1774062167714436374.jpeg`。
- 同一 segment 内图片应具有相同宽高和宽高比。
- 不建议把 5698 张全量 30 FPS 图像一次性作为一个 scene 输入。
- 对 24GB 单卡，建议先用 2-8 帧 smoke test，再使用 8-32 帧常规 segment；更大帧数需要根据显存逐步放大。

WorldMirror 预处理会 resize 到 `target_size` 附近，并按中心裁剪到 patch size 对齐的尺寸。输入内参应仍然对应原始图像分辨率，pipeline 会自动做 resize + center crop 的内参调整。

### 4.2 相机先验约束

WorldMirror `prior_cam_path` 的 `camera_params.json` 格式：

```json
{
  "num_cameras": 2,
  "extrinsics": [
    {
      "camera_id": "1774062167714436374",
      "matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ],
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

关键规则：

- `camera_id` 可以是图片序号，也可以是图片文件 stem；本数据集推荐使用 stem。
- `extrinsics.matrix` 是 4x4 camera-to-world，OpenCV camera convention。
- `intrinsics.matrix` 是原始图像分辨率下的 3x3 K。
- `extrinsics` 和 `intrinsics` 可以独立提供。
- 如果某类 prior 的匹配数量不是当前输入图片总数 N，WorldMirror 会禁用该类 prior。
- 因此 `camera_params.json` 不能只给部分帧内参或部分帧外参。

## 5. 分阶段制作流程

### Phase 0：确定输入和制作配置

输入参数建议显式记录到 `manifest.json`：

```json
{
  "dataset_version": "mono_worldmirror_v0.2",
  "source_track_dir": "/media/haiming/data2/2026_test/0521_ramp/7869148/pipeline/public/MX11-2S/HXMQBC7PZ4CT25SL1/2026-03-21/110247/110557/644470719",
  "track_id": "644470719",
  "camera_name": "mid_center_top_wide",
  "pose_source": "deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json",
  "image_source": "camera/mid_center_top_wide",
  "segment_policy": {
    "use_pose_keyframes_only": true,
    "max_gap_s": 1.0,
    "min_frames": 2,
    "max_frames": 32
  },
  "camera_prior_policy": {
    "default": "intrinsics_only",
    "full_extrinsics_enabled": false,
    "reason": "pose.trans appears to be longitude/latitude/height in the verified sample"
  },
  "image_materialization": "symlink"
}
```

制作工具必须把这个配置随数据集一起保存，避免后续不知道数据是如何生成的。

### Phase 1：输入完整性审计

检查以下路径：

```text
<single_track_dir>/calib.json
<single_track_dir>/camera/mid_center_top_wide/
<single_track_dir>/deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json
```

审计项：

| 项 | 要求 |
| --- | --- |
| `calib.json` | 文件存在、JSON 可解析、包含目标相机 |
| 图像目录 | 存在，至少有 2 张可读取图片 |
| 图像命名 | 文件 stem 可转成纳秒时间戳整数 |
| 图像格式 | JPEG/PNG/WebP 可被 PIL/OpenCV 读取 |
| 图像尺寸 | 同一相机目录中分辨率一致或差异被明确记录 |
| 位姿文件 | JSON 可解析，数组非空 |
| 位姿时间戳 | `timestamp` 或 `name` 可与图像 stem 匹配 |

审计脚本示例：

```bash
python - <<'PY'
import json
from pathlib import Path
from PIL import Image

root = Path("/media/haiming/data2/2026_test/0521_ramp/7869148/pipeline/public/MX11-2S/HXMQBC7PZ4CT25SL1/2026-03-21/110247/110557/644470719")
img_dir = root / "camera/mid_center_top_wide"
pose_path = root / "deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json"

img_paths = sorted(img_dir.glob("*.jpeg"))
poses = json.load(open(pose_path))
with Image.open(img_paths[0]) as im:
    first_size = im.size

img_ts = [int(p.stem) for p in img_paths]
pose_ts = [int(p["timestamp"]) for p in poses]

print("images:", len(img_paths), img_ts[0], img_ts[-1], first_size)
print("poses:", len(poses), pose_ts[0], pose_ts[-1])
print("exact matches:", sum(str(t) in {p.stem for p in img_paths} for t in pose_ts))
PY
```

把审计结果写入：

```text
raw_audit.json
timestamp_match_report.json
```

### Phase 2：抽取目标相机标定

从 `calib.json` 的 `cameras` 列表中找到：

```text
name.label == SENSOR_LABEL_MID_CENTER_TOP_WIDE
```

当前样例内参：

```text
fx = 1904.99609375
fy = 1903.804931640625
cx = 1917.16064453125
cy = 1076.6253662109375
```

当前样例 KB 畸变：

```text
[-0.029459979385137558, -0.005299502518028021, -0.0012424745364114642, 0.0012543649645522237]
```

需要保存的信息：

- `camera_name`
- `sensor_label`
- `image_width`
- `image_height`
- `camera_model`
- `intrinsics`
- `distortion`
- `sensor_to_vehicle`
- `sensor_to_vehicle_eol`
- `serial_number`
- `update_timestamp`
- `status`

建议保存为：

```text
source_calib.json
```

只保存目标相机条目，避免下游误用其它相机标定。

### Phase 3：图像与位姿时间戳匹配

匹配规则优先级：

1. `str(pose["timestamp"]) == image_path.stem`
2. `pose["name"] == image_path.stem`
3. 如确实没有 exact match，才允许使用最近邻匹配，并把误差写入报告。

当前样例不需要最近邻匹配，因为 `191/191` exact match。

输出报告建议包含：

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

验收规则：

- 使用 camera prior 的 segment：segment 内每张图片必须有位姿和内参记录。
- 不使用 camera prior 的 segment：可以只要求图片存在，但仍建议保存匹配报告。
- 最近邻匹配误差如果大于 5 ms，应作为风险记录；大于单帧周期一半时，不建议自动匹配。

### Phase 4：关键帧选择

推荐第一版只使用 SFM 优化关键帧图像：

```text
selected_images = images whose stem is in optimized_keyframe_poses.json
```

原因：

- SFM 优化位姿只有 191 条，和关键帧一一对应。
- 全量 5698 张图像跨越约 190 秒，单次前馈输入过大。
- 如果把非关键帧加入 segment，camera prior 无法覆盖全部输入帧，会导致 prior 被 WorldMirror 禁用。

可选扩展：

| 策略 | 适用场景 | 注意事项 |
| --- | --- | --- |
| 只用 SFM 关键帧 | 第一版推理数据集、外参对比、快速 QA | 推荐默认 |
| 按固定 FPS 从全量图像采样 | 不使用外参 prior，只想看模型重建效果 | 需要记录采样规则 |
| 用插值位姿补齐非关键帧 | 想构造更密集的有先验序列 | 必须验证插值轨迹和坐标系 |
| 使用视频输入 | 快速体验 WorldMirror | 不利于追溯逐帧 pose/calib |

### Phase 5：按时间连续性切 segment

对已匹配关键帧按时间戳排序，若相邻关键帧间隔超过阈值，则开启新 segment。

推荐参数：

| 参数 | 建议值 | 说明 |
| --- | ---: | --- |
| `max_gap_s_strict` | 1.0 | 第一版保守切分 |
| `max_gap_s_tolerant` | 2.0 | 如果 segment 太碎，可以用该值 |
| `min_frames` | 2 | WorldMirror 可单图，但 3DGS/相机估计更建议多图 |
| `smoke_test_frames` | 2-8 | 首次跑通链路 |
| `regular_frames` | 8-32 | 24GB 单卡较稳妥的起点 |
| `large_frames` | 32-64 | 需要按显存和 `target_size` 验证 |

切段输出 `segments.json` 示例：

```json
{
  "policy": {
    "use_pose_keyframes_only": true,
    "max_gap_s": 1.0,
    "min_frames": 2,
    "max_frames": 32
  },
  "segments": [
    {
      "segment_id": "seg_000",
      "frame_count": 8,
      "start_timestamp": "1774062167714436374",
      "end_timestamp": "1774062169947760307",
      "max_gap_s": 0.266667,
      "source": "optimized_keyframe_poses"
    }
  ],
  "dropped_segments": [
    {
      "reason": "frame_count_less_than_min_frames",
      "frame_count": 1,
      "timestamps": ["..."]
    }
  ]
}
```

不要把跨越十几秒或几十秒断裂的关键帧硬塞进同一个 segment。WorldMirror 是前馈多视图重建模型，输入应尽量表达同一局部连续场景。

### Phase 6：图像落盘

推荐默认使用软链接：

```bash
ln -s <single_track_dir>/camera/mid_center_top_wide/1774062167714436374.jpeg \
      <segment_dir>/images/1774062167714436374.jpeg
```

策略选择：

| 策略 | 优点 | 缺点 |
| --- | --- | --- |
| symlink | 节省空间，制作快 | 数据集迁移时容易断链 |
| hardlink | 节省空间，迁移同磁盘较稳 | 跨文件系统不可用 |
| copy | 数据集自包含 | 占空间，制作慢 |
| rectify-copy | 几何更规范 | 需要重新计算 K，且要保存处理记录 |

要求：

- 不要默认重命名图片。
- 如果必须重命名为 `000000.jpeg` 这类序号，必须同步改写 `camera_id`，并保留 `timestamps.txt` 的原始时间戳映射。
- 如果做 undistort/rectify，必须记录原始 K、畸变参数、新 K、输出分辨率、有效 ROI。

### Phase 7：内参 prior 生成

只注入内参时，生成 `camera_params_intrinsics_only.json`：

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
    },
    {
      "camera_id": "1774062167947755496",
      "matrix": [
        [1904.99609375, 0.0, 1917.16064453125],
        [0.0, 1903.804931640625, 1076.6253662109375],
        [0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

验收规则：

- `num_cameras == len(images)`。
- `len(intrinsics) == len(images)`。
- 每个 `camera_id` 都能匹配当前 `images/` 里的一个 stem。
- K 对应输入原始分辨率。如果图像做过去畸变或缩放，必须使用处理后的 K。

当前样例是 KB/Kannala-Brandt 广角模型，WorldMirror prior 只表达 pinhole K。策略如下：

| 策略 | 建议 |
| --- | --- |
| 原图 + pinhole K | 可用于链路 smoke test，但几何精度有风险 |
| 去畸变图 + 新 pinhole K | 正式数据集推荐 |
| 扩展 WorldMirror 相机模型 | 工作量较大，不作为第一版 |

如果使用去畸变图，`meta.json` 需要记录：

```json
{
  "image_processing": {
    "undistorted": true,
    "source_camera_model": "CAMERA_MODEL_KANNEL_BRAND",
    "target_camera_model": "pinhole",
    "source_resolution": [3840, 2160],
    "output_resolution": [3840, 2160],
    "intrinsics_source": "source_calib.json",
    "intrinsics_used": "camera_params_intrinsics_only.json"
  }
}
```

### Phase 8：外参 prior 生成，默认关闭

WorldMirror 要求：

```text
extrinsics.matrix = camera-to-world 4x4, OpenCV camera convention
```

当前样例不能直接生成可信外参，因为：

- `pose.trans` 数值接近经纬高。
- `optimized_keyframe_poses.json` 中 `quat` 的语义和顺序需要确认。
- `sensor_to_vehicle` / `sensor_to_vehicle_eol` 是否可直接作为 camera-to-vehicle 需要确认。
- vehicle/world 坐标和 OpenCV camera 坐标轴的转换需要明确。

正式开启外参 prior 前，需要完成以下验证：

1. 确认 `pose.trans` 是经纬高还是局部米制坐标。
2. 如果是经纬高，选择 ENU/UTM/local tangent plane 原点，并转换成米制。
3. 确认 `quat` 表示 `vehicle_to_world` 还是 `world_to_vehicle`。
4. 确认 `quat` 顺序是 `wxyz` 还是 `xyzw`。
5. 确认 `sensor_to_vehicle` 的方向，是否为 `T_camera_to_vehicle`。
6. 确认相机坐标轴是否已经是 OpenCV convention。
7. 用相机 frustum 可视化检查轨迹朝向和相邻帧平滑性。

候选组合关系：

```text
T_camera_to_world = T_vehicle_to_world @ T_camera_to_vehicle
```

如果标定实际是反方向，则需要求逆。这个关系不能只凭字段名确定，必须结合标定文档或可视化验证。

通过验证后，才生成 `camera_params_full.json`：

```json
{
  "num_cameras": 2,
  "extrinsics": [
    {
      "camera_id": "1774062167714436374",
      "matrix": [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]
    },
    {
      "camera_id": "1774062167947755496",
      "matrix": [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]
    }
  ],
  "intrinsics": [
    {
      "camera_id": "1774062167714436374",
      "matrix": [[1904.99609375, 0.0, 1917.16064453125], [0.0, 1903.804931640625, 1076.6253662109375], [0.0, 0.0, 1.0]]
    }
  ]
}
```

当前样例的默认 `camera_params.json` 应指向 intrinsics-only 版本：

```text
camera_params.json -> camera_params_intrinsics_only.json
```

### Phase 9：segment 元数据和质检报告

每个 segment 的 `meta.json` 推荐格式：

```json
{
  "dataset_version": "mono_worldmirror_v0.2",
  "track_id": "644470719",
  "camera_name": "mid_center_top_wide",
  "segment_id": "seg_000",
  "source_track_dir": "/media/haiming/data2/2026_test/0521_ramp/7869148/pipeline/public/MX11-2S/HXMQBC7PZ4CT25SL1/2026-03-21/110247/110557/644470719",
  "image_source": "camera/mid_center_top_wide",
  "pose_source": "deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json",
  "frame_count": 8,
  "timestamp_start": "1774062167714436374",
  "timestamp_end": "1774062169947760307",
  "image_resolution": [3840, 2160],
  "image_materialization": "symlink",
  "camera_prior": {
    "used_file": "camera_params.json",
    "mode": "intrinsics_only",
    "extrinsics_enabled": false
  },
  "segmentation_policy": {
    "max_gap_s": 1.0,
    "min_frames": 2,
    "max_frames": 32
  }
}
```

`qc_report.json` 推荐格式：

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

### Phase 10：运行 WorldMirror 推理

不注入相机先验：

```bash
conda run -n hyworld2 bash -lc '
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=0 python -m hyworld2.worldrecon.pipeline \
  --input_path <segment_dir>/images \
  --strict_output_path <output_dir> \
  --target_size 392 \
  --enable_bf16 \
  --no_sky_mask \
  --no_edge_mask \
  --no_interactive
'
```

注入 intrinsics-only prior：

```bash
conda run -n hyworld2 bash -lc '
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=0 python -m hyworld2.worldrecon.pipeline \
  --input_path <segment_dir>/images \
  --prior_cam_path <segment_dir>/camera_params.json \
  --strict_output_path <output_dir> \
  --target_size 392 \
  --enable_bf16 \
  --no_sky_mask \
  --no_edge_mask \
  --no_interactive
'
```

参数建议：

| 参数 | 建议 |
| --- | --- |
| `--target_size 392` | 4090D/24GB 上做 smoke test 更稳 |
| `--target_size 518/672/952` | 逐步放大验证质量和显存 |
| `--enable_bf16` | 可降低显存压力 |
| `--no_sky_mask --no_edge_mask` | 首次复现链路可关闭，避免额外依赖影响排查 |
| `--strict_output_path` | 推荐使用，便于把输出绑定到 segment |

成功输出至少应包含：

```text
camera_params.json
depth/depth_*.npy
depth/depth_*.png
normal/normal_*.png
points.ply
gaussians.ply
pipeline_timing.json
```

如果 `gaussians.ply` 不生成，检查：

- 是否传入了 `--no_save_gs`。
- 是否 `--disable_heads` 包含 `gs`。
- 是否推理中途 OOM 或异常退出。
- 输出目录是否被旧文件污染。

### Phase 11：输出质检和可视化

输出质检：

| 输出 | 检查方法 | 预期 |
| --- | --- | --- |
| `camera_params.json` | JSON 解析，检查 `camera_poses`/intrinsics 数量 | 数量等于输入帧数 |
| `depth/depth_*.npy` | `numpy.load` 检查 shape、NaN、范围 | 每帧一份 |
| `depth/depth_*.png` | 图片查看器或 OpenCV | 可视化深度连续 |
| `normal/normal_*.png` | 图片查看器 | 法线方向无明显全黑/全白 |
| `points.ply` | CloudCompare/Open3D/Rerun | 点云能打开，尺度合理 |
| `gaussians.ply` | SuperSplat/gsplat viewer | 3DGS 能加载渲染 |
| `pipeline_timing.json` | JSON 解析 | 记录推理耗时 |

推荐可视化工具：

- `gaussians.ply`：SuperSplat，适合快速查看 3DGS 渲染效果。
- `points.ply`：CloudCompare 或 Open3D，适合检查点云尺度、稀疏/漂移/离群点。
- `camera_params.json + points/depth`：Rerun，适合把相机轨迹、点云、深度、法线放在同一时间轴中调试。
- `depth/normal`：Python/matplotlib、Rerun 或普通图片查看器。

外参 prior 开启后必须做额外检查：

1. 画 camera frustum，确认相机朝向和车辆运动方向一致。
2. 检查相邻帧相机轨迹是否平滑。
3. 检查平移单位是否是米。
4. 检查第一帧归一化后相对轨迹是否合理。
5. 将深度反投影点云和 WorldMirror `points.ply` 对比。

### Phase 12：数据集总 manifest

`manifest.json` 示例：

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
    "extrinsics_enabled": false
  },
  "segments": [
    {
      "track_id": "644470719",
      "camera_name": "mid_center_top_wide",
      "segment_id": "seg_000",
      "relative_path": "tracks/644470719/mid_center_top_wide/seg_000",
      "frame_count": 8,
      "camera_prior": "intrinsics_only"
    }
  ]
}
```

manifest 是后续批量推理、批量可视化和结果回溯的入口，不建议省略。

## 6. 自动化脚本建议

虽然本文只定义流程，但后续实现脚本时建议拆成以下函数：

```python
def audit_inputs(track_dir: Path, camera_name: str, pose_relpath: str) -> dict:
    ...

def load_camera_calib(calib_path: Path, camera_name: str) -> dict:
    ...

def load_image_index(image_dir: Path) -> dict[str, Path]:
    ...

def load_optimized_poses(pose_path: Path) -> list[dict]:
    ...

def match_poses_to_images(poses: list[dict], image_by_stem: dict[str, Path]) -> list[dict]:
    ...

def split_segments(matches: list[dict], max_gap_s: float, min_frames: int, max_frames: int) -> list[dict]:
    ...

def materialize_images(segment: dict, out_image_dir: Path, mode: str) -> None:
    ...

def write_intrinsics_only_camera_params(segment: dict, calib: dict, out_path: Path) -> None:
    ...

def write_segment_metadata(segment: dict, out_dir: Path) -> None:
    ...

def write_manifest(dataset_root: Path, tracks: list[dict]) -> None:
    ...
```

主流程伪代码：

```python
root = Path(single_track_dir)
camera_name = "mid_center_top_wide"
pose_relpath = "deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json"

audit = audit_inputs(root, camera_name, pose_relpath)
calib = load_camera_calib(root / "calib.json", camera_name)
poses = load_optimized_poses(root / pose_relpath)
image_by_stem = load_image_index(root / "camera" / camera_name)

matches = match_poses_to_images(poses, image_by_stem)
segments = split_segments(matches, max_gap_s=1.0, min_frames=2, max_frames=32)

track_out = dataset_root / "tracks" / "644470719" / camera_name
save_json(track_out / "raw_audit.json", audit)
save_json(track_out / "source_calib.json", calib)
save_json(track_out / "source_pose_all.json", poses)
save_json(track_out / "timestamp_match_report.json", build_match_report(matches))
save_json(track_out / "segments.json", build_segments_report(segments))

for segment in segments:
    seg_out = track_out / segment["segment_id"]
    materialize_images(segment, seg_out / "images", mode="symlink")
    save_json(seg_out / "source_poses.json", segment["poses"])
    save_text(seg_out / "timestamps.txt", "\n".join(segment["timestamps"]) + "\n")
    write_intrinsics_only_camera_params(segment, calib, seg_out / "camera_params_intrinsics_only.json")
    link_or_copy(seg_out / "camera_params_intrinsics_only.json", seg_out / "camera_params.json")
    write_segment_metadata(segment, seg_out)
    write_qc_report(segment, seg_out)

write_manifest(dataset_root, tracks=[...])
```

## 7. 验收清单

数据集制作完成后，逐项检查：

| 检查项 | 通过条件 |
| --- | --- |
| 原始路径记录 | `manifest.json/raw_audit.json/meta.json` 中都有原始单程路径 |
| 标定抽取 | `source_calib.json` 只包含目标相机，内参和分辨率正确 |
| 图像可读 | 每个 `images/` 中图片存在且 PIL/OpenCV 可读取 |
| 时间戳匹配 | 使用 SFM 关键帧时，segment 内图片全部 exact match |
| segment 连续性 | 相邻帧最大间隔不超过策略阈值，或被明确标记 |
| frame count | 每个 segment 满足 `min_frames <= N <= max_frames` |
| prior 覆盖 | 使用 `camera_params.json` 时，intrinsics/extrinsics 覆盖全部 N 帧 |
| camera_id | `camera_id` 与图片 stem 一一对应 |
| 外参安全 | 坐标语义未确认时，不启用 extrinsics |
| 图像处理记录 | 做过去畸变/缩放/裁剪时，记录新 K 和处理参数 |
| WorldMirror 输出 | `camera_params.json/depth/normal/points.ply/gaussians.ply` 按预期生成 |
| 可视化检查 | 至少能打开 `points.ply` 和 `gaussians.ply` |

一票否决项：

- `camera_id` 和图片 stem 对不上。
- `camera_params.json` 只覆盖部分输入帧。
- 把经纬高直接写成 c2w 平移。
- 切段跨越十几秒或几十秒但未记录原因。
- 使用去畸变图却仍写原始 K。

## 8. 常见问题和排查

| 问题 | 常见原因 | 处理建议 |
| --- | --- | --- |
| WorldMirror 打印 prior matched 小于 N | `camera_id` 与 stem 不一致，或 prior 只写了部分帧 | 用图片 stem 作为 `camera_id`，并保证每帧都有记录 |
| 输出重建尺度/轨迹明显错 | 外参单位或坐标轴错误 | 关闭 extrinsics，仅用 intrinsics-only 对比 |
| 点云很散或重影 | segment 跨越太大、动态物体多、畸变未处理 | 缩短 segment，做去畸变，先用连续 8-16 帧 |
| `gaussians.ply` 缺失 | 禁用了 gs head、OOM、中途异常 | 检查命令参数、日志和显存，降低 `target_size/frame_count` |
| depth/normal 数量少于输入帧 | 推理失败或输出目录被污染 | 清空输出目录后重跑，检查日志 |
| 图像读取报分辨率不一致 | 混入不同尺寸图片或处理后输出尺寸不统一 | 每个 segment 只保留同一处理链路下的图片 |
| 相机朝向反了 | `T_vehicle_to_world` / `T_world_to_vehicle` 或 sensor 外参方向弄反 | 可视化 frustum，逐项测试矩阵求逆 |
| 轨迹平移量极小但经纬变化明显 | 经纬高未转米制 | 使用 ENU/UTM/local tangent plane 转换 |
| 注入内参后结果变差 | KB 广角畸变被当成 pinhole | 使用去畸变图和新 K，或先不注入 prior |

## 9. 当前样例推荐落地方案

对当前样例，推荐按以下顺序制作：

1. 以 `deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json` 作为权威 SFM 位姿来源。
2. 从 `camera/mid_center_top_wide` 中抽取 191 张 exact match 的关键帧图像。
3. 用 `max_gap_s=1.0` 做保守切段；如果 segment 太碎，再评估 `max_gap_s=2.0`。
4. 每个 segment 先限制在 2-32 帧，先做 2-8 帧 smoke test。
5. 生成 `source_calib.json/source_poses.json/timestamps.txt/meta.json/qc_report.json`。
6. 先生成 intrinsics-only `camera_params.json`，不要生成默认启用的 full extrinsics。
7. 用 `--target_size 392 --enable_bf16 --no_sky_mask --no_edge_mask` 跑通 WorldMirror。
8. 使用 Rerun/Open3D/CloudCompare/SuperSplat 做输出检查。
9. 再单独研究 `pose.trans/quat/sensor_to_vehicle` 的坐标语义，确认后再生成 `camera_params_full.json`。

这样可以先稳定复现 WorldMirror 前馈式 3DGS 输出，同时避免错误外参把重建结果带偏。

## 10. 和其它文档的关系

相关文档：

```text
docs/data_format/dir_structure/单程目录结构.md
docs/data_format/dir_structure/mono_dataformat.md
docs/view/可视化工具选取建议.md
DOCUMENTATION.md
hyworld2/worldrecon/hyworldmirror/utils/inference_utils.py
```

分工：

- `单程目录结构.md`：说明原始单程目录里有什么。
- `mono_dataformat.md`：说明当前单目样例的数据格式、字段、标定和匹配分析。
- `mono_dataset_generation.md`：说明如何把原始单程制作成可推理数据集，也就是本文。
- `docs/view/可视化工具选取建议.md`：说明输出产物如何可视化和调试。
- `DOCUMENTATION.md`：WorldMirror 官方接口和输出格式说明。
- `inference_utils.py`：WorldMirror 实际读取图像、读取 prior、匹配 `camera_id` 的代码约束。
