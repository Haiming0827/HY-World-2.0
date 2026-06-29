# Segmentation and Mask Generation for Mono WorldMirror Segments

本文分析如何把
`dataset/tracks/644470719/mid_center_top_wide/seg_000`、`seg_001` 等分段目录中的图像生成 Mask2Former 语义分割结果，并进一步为 HY-World WorldMirror 准备天空和边缘相关 mask。

## 1. 结论

1. `Mask2Former` 的 `16: sky` 可以作为外部天空语义来源，用来生成 WorldMirror 口径的 `sky_mask`。
2. `31: ego_mask` 不能等同于 WorldMirror 的 `edge_mask`。`ego_mask` 是车体或自车遮挡区域语义；WorldMirror 的 `edge_mask` 是深度或法线不连续边缘过滤，代码中由预测 depth/normal 计算。
3. 如果只依赖 Mask2Former 分割结果，可从语义 mask 额外生成一个 `semantic_edge_mask`，用于调试或后续扩展；但它只能近似“语义边界”，不是当前 WorldMirror 内部 `apply_edge_mask` 的完全替代。
4. 当前 `seg_xxx/camera_params.json` 只包含每帧 3x3 内参，没有畸变参数。真正做 KB 鱼眼去畸变需要父目录的 `source_calib.json`，或原始单程目录中的 `calib.json`。如果只读取 `camera_params.json`，只能得到 K，不能完整执行去畸变。

## 2. 现有依据

### 2.1 Mask2Former 工程用法

`/home/haiming/admap_ws/seg_mask/docs/README.md` 中记录的 V1 pipeline：

```text
/home/haiming/admap_ws/seg_mask/mask2former/pipelines/run_mask2former_pipeline.sh
/home/haiming/admap_ws/seg_mask/mask2former/pipelines/infer_masks.py
```

默认权重：

```text
/home/haiming/admap_ws/seg_mask/mask2former/weights/mask2former_r50_cadata_1920x1080_90000.pt
```

默认输出结构是单程根目录下的：

```text
<single_track_dir>/deep-learning/vision-map/<run_id>/results/mask2former/
├── manifest.json
├── class_labels.json
├── missing_images.txt
├── mask_stats.jsonl
├── image_segment/
│   └── mid_center_top_wide/
│       └── <timestamp_ns>.png
├── image_undistort/
│   └── mid_center_top_wide/
│       └── <timestamp_ns>.jpeg
└── color_viz/
    └── mid_center_top_wide/
        └── <timestamp_ns>.png
```

其中 `image_segment/<camera>/*.png` 是单通道语义 mask，像素值就是类别 ID。

代码中的类别表确认：

```text
16: sky
31: ego_mask
```

`infer_masks.py` 还会在推理前做去畸变，并保存 `image_undistort`。现有去畸变逻辑使用 `cv2.fisheye.initUndistortRectifyMap`，需要 K 和 `distortions_k1..k4`。

### 2.2 当前 mono segment 数据

目标目录：

```text
/home/haiming/open_source/world_models/HY-World-2.0/dataset/tracks/644470719/mid_center_top_wide/
```

当前物化出的有效分段为 `seg_000` 到 `seg_007`。每段有：

```text
seg_xxx/
├── images/
│   └── <timestamp_ns>.jpeg -> <source_track>/camera/mid_center_top_wide/<timestamp_ns>.jpeg
├── camera_params.json
├── camera_params_intrinsics_only.json
├── source_poses.json
├── timestamps.txt
├── meta.json
└── qc_report.json
```

当前检查到每段 `images/` 里的图像软链数与 `timestamps.txt` 行数一致：

| Segment | Frames |
| --- | ---: |
| `seg_000` | 32 |
| `seg_001` | 9 |
| `seg_002` | 11 |
| `seg_003` | 18 |
| `seg_004` | 15 |
| `seg_005` | 32 |
| `seg_006` | 31 |
| `seg_007` | 32 |

注意：父目录 `segments.json` 仍包含被过滤掉的短片段索引，因此它的 `seg_002` 之后编号和当前已物化目录不完全一致。后续生成 mask 时应以每个 `seg_xxx/meta.json`、`timestamps.txt`、`images/` 和 `camera_params.json` 的一致性为准。

### 2.3 标定和去畸变

`seg_xxx/camera_params.json` 当前是 WorldMirror camera prior，内容为：

```json
{
  "num_cameras": 32,
  "extrinsics": [],
  "intrinsics": [
    {
      "camera_id": "1774062167714436374",
      "matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    }
  ]
}
```

它没有畸变参数。父目录 `source_calib.json` 才保存：

```json
{
  "camera_model": "CAMERA_MODEL_KANNEL_BRAND",
  "intrinsics": {"fx": 1904.99609375, "fy": 1903.804931640625, "cx": 1917.16064453125, "cy": 1076.6253662109375},
  "distortion": {"k1": -0.029459979385137558, "k2": -0.005299502518028021, "k3": -0.0012424745364114642, "k4": 0.0012543649645522237}
}
```

因此推荐的输入规则是：

1. 帧列表和每帧 K：读取 `seg_xxx/timestamps.txt` 和 `seg_xxx/camera_params.json`。
2. 去畸变所需 KB 畸变：读取 `../source_calib.json`。
3. 如果未来 `camera_params.json` 扩展出 `distortion` 字段，可以优先使用 segment 内文件；在当前样例里不能只依赖它。

## 3. WorldMirror mask 语义

WorldMirror pipeline 的相关参数：

```text
--apply_sky_mask / --no_sky_mask
--apply_edge_mask / --no_edge_mask
--save_sky_mask
--sky_mask_source auto|model|onnx
```

代码口径：

1. `compute_sky_mask(...)` 返回 `[S,H,W] bool`，语义是 `True = non-sky / keep`，`False = sky / filter out`。
2. `save_sky_mask=True` 时，保存 PNG 前会做 `~sky_mask`，所以落盘的 `sky_mask/sky_mask_0000.png` 是 `255 = sky`，`0 = non-sky`。
3. `apply_edge_mask=True` 时，`create_filter_mask(...)` 先用 `depth_edge(...)` 和 `normals_edge(...)` 找不连续边缘，再取反得到保留区域：`edge_mask = ~combined_edges`。
4. 最终用于点云或 Gaussian 过滤的 mask 是 `confidence_mask & edge_keep_mask & sky_keep_mask`，语义是 `True = keep`。

因此：

| 来源 | 可否对应 WorldMirror | 说明 |
| --- | --- | --- |
| `segment == 16` sky | 可以生成外部 `sky_mask` | 分割 PNG 中 `16` 为天空；保存为独立二值 PNG 时建议 `255=sky`，与 WorldMirror `save_sky_mask` 的落盘口径一致 |
| `segment == 31` ego_mask | 不等于 `edge_mask` | 这是自车遮挡语义，适合生成 `ego_mask` 或额外 invalid 区域 |
| 语义边界 | 可生成 `semantic_edge_mask` | 可由分割类别变化、`sky` 边界、`ego_mask` 边界膨胀得到；只表示语义边界，不是 depth/normal edge |
| WorldMirror internal edge | 不能仅从 Mask2Former 直接得到 | 需要 WorldMirror 预测的 depth/normal 或其它几何信号 |

## 4. 推荐输出结构

对每个 `seg_xxx`，推荐直接在 segment 目录下生成以下结构：

```text
seg_xxx/
├── images/
├── camera_params.json
├── image_segment/
│   ├── manifest.json
│   ├── class_labels.json
│   ├── mask_stats.jsonl
│   ├── semantic/
│   │   └── <timestamp_ns>.png
│   ├── undistorted/
│   │   └── <timestamp_ns>.jpeg
│   └── color_viz/
│       └── <timestamp_ns>.png
├── masks/
│   ├── sky_mask/
│   │   └── <timestamp_ns>.png
│   ├── ego_mask/
│   │   └── <timestamp_ns>.png
│   ├── semantic_edge_mask/
│   │   └── <timestamp_ns>.png
│   ├── worldmirror_keep_mask/
│   │   └── <timestamp_ns>.png
│   └── manifest.json
└── ...
```

文件语义：

| 路径 | 格式 | 像素语义 |
| --- | --- | --- |
| `image_segment/semantic/<timestamp_ns>.png` | `uint8` 单通道 PNG | Mask2Former 类别 ID，`16=sky`，`31=ego_mask` |
| `image_segment/undistorted/<timestamp_ns>.jpeg` | RGB/BGR 图像文件 | 本次分割实际使用的去畸变图像 |
| `image_segment/color_viz/<timestamp_ns>.png` | 彩色 PNG | 调试预览，不作为算法输入 |
| `masks/sky_mask/<timestamp_ns>.png` | `uint8` 单通道 PNG | `255=sky`，`0=non-sky` |
| `masks/ego_mask/<timestamp_ns>.png` | `uint8` 单通道 PNG | `255=ego/self vehicle`，`0=other` |
| `masks/semantic_edge_mask/<timestamp_ns>.png` | `uint8` 单通道 PNG | `255=semantic boundary / invalid edge candidate`，`0=non-edge` |
| `masks/worldmirror_keep_mask/<timestamp_ns>.png` | `uint8` 单通道 PNG | `255=keep`，`0=filter out` |

命名建议使用原始时间戳 `<timestamp_ns>`，不要改成 `sky_mask_0000.png`，原因是：

1. segment 输入图像、`camera_params.json` 的 `camera_id`、`timestamps.txt` 都以时间戳对齐。
2. 时间戳命名便于跨目录 join，不依赖帧序号是否重新排序。
3. 如果需要模拟 WorldMirror 自带 `save_sky_mask` 产物，可以额外生成 `worldmirror_sky_mask/sky_mask_<idx:04d>.png`，但不应作为主索引。

## 5. 生成规则

### 5.1 语义分割

每个 segment 的输入是：

```text
seg_xxx/images/<timestamp_ns>.jpeg
seg_xxx/timestamps.txt
seg_xxx/camera_params.json
../source_calib.json
```

处理步骤：

1. 按 `timestamps.txt` 顺序收集图像，并校验每个 stem 都出现在 `camera_params.json.intrinsics[*].camera_id`。
2. 从 `camera_params.json` 读取每帧 K；如果每帧 K 相同，可以复用同一 map。
3. 从 `../source_calib.json` 读取 `k1..k4`。
4. 用 OpenCV fisheye/KB 模型去畸变图像。
5. 用 Mask2Former 39 类权重推理去畸变图像。
6. 保存语义类别 ID PNG 到 `image_segment/semantic/<timestamp_ns>.png`。
7. 保存去畸变图像到 `image_segment/undistorted/<timestamp_ns>.jpeg`。
8. 写 `image_segment/manifest.json`、`class_labels.json`、`mask_stats.jsonl`。

现有 `/home/haiming/admap_ws/seg_mask/mask2former/pipelines/infer_masks.py` 的核心推理、类别表、resize/pad/decode、去畸变逻辑可复用。但它当前入口面向“单程根目录 + deep-learning/vision-map/<run_id>”，会去找 `<track_dir>/camera/<camera_name>` 和 `<track_dir>/calib.json`。对当前 `seg_xxx` 数据，建议新增轻量 segment adapter，而不是直接把 `seg_xxx` 当成旧单程根目录。

### 5.2 从语义分割生成 sky mask

从 `image_segment/semantic/<timestamp_ns>.png` 读取 `seg`：

```text
sky_binary = (seg == 16)
```

落盘：

```text
masks/sky_mask/<timestamp_ns>.png = sky_binary * 255
```

这个落盘极性与 WorldMirror `save_sky_mask` 的 PNG 一致：`255=sky`。如果要把它传回 WorldMirror 内部的 filter 逻辑，内存中需要转换成：

```text
sky_keep_mask = (seg != 16)
```

也就是 `True=non-sky/keep`。

### 5.3 从语义分割生成 ego mask

从 `seg` 读取：

```text
ego_binary = (seg == 31)
```

落盘：

```text
masks/ego_mask/<timestamp_ns>.png = ego_binary * 255
```

`ego_mask` 可用于排除自车遮挡区域。若要合入最终 keep mask：

```text
ego_keep_mask = (seg != 31)
```

### 5.4 生成 semantic edge mask

如果没有 WorldMirror depth/normal 预测，只能从语义图近似边界：

```text
edge = boundary(seg)
edge = dilate(edge, kernel=3 or 5)
edge |= dilate(seg == 16, kernel=3) boundary band
edge |= dilate(seg == 31, kernel=3 or 5)
```

推荐落盘：

```text
masks/semantic_edge_mask/<timestamp_ns>.png = edge * 255
```

其中 `255=语义边界或应过滤候选`。这类 mask 可以用于调试和后处理，但文档层面不建议称它为 WorldMirror 原生 `edge_mask`，避免和 depth/normal discontinuity 混淆。

### 5.5 生成 worldmirror_keep_mask

如果暂时只用外部分割结果构造过滤 mask，推荐：

```text
keep = (seg != 16) & (seg != 31) & (~semantic_edge)
masks/worldmirror_keep_mask/<timestamp_ns>.png = keep * 255
```

其中 `255=keep`，`0=filter out`。它的布尔语义对齐 WorldMirror 内部 `filter_mask`，但它不是当前 `pipeline.py` 已有参数能直接读取的输入文件。若要真正参与 WorldMirror 过滤，需要后续扩展 pipeline，让 `compute_filter_mask(...)` 接受外部 keep mask 或外部 sky mask。

## 6. Manifest 建议

`image_segment/manifest.json` 建议记录：

```json
{
  "pipeline": "mask2former_segment_v1",
  "segment_dir": "seg_000",
  "camera_name": "mid_center_top_wide",
  "input_images": "images",
  "camera_params": "camera_params.json",
  "calibration_source": "../source_calib.json",
  "undistort": true,
  "model_path": "/home/haiming/admap_ws/seg_mask/mask2former/weights/mask2former_r50_cadata_1920x1080_90000.pt",
  "class_labels": "class_labels.json",
  "outputs": {
    "semantic_dir": "image_segment/semantic",
    "undistorted_dir": "image_segment/undistorted",
    "color_viz_dir": "image_segment/color_viz"
  },
  "frames": [
    {
      "timestamp_ns": "1774062167714436374",
      "image": "images/1774062167714436374.jpeg",
      "semantic": "image_segment/semantic/1774062167714436374.png",
      "undistorted": "image_segment/undistorted/1774062167714436374.jpeg"
    }
  ]
}
```

`masks/manifest.json` 建议记录：

```json
{
  "pipeline": "semantic_masks_from_mask2former_v1",
  "semantic_source": "image_segment/semantic",
  "class_ids": {
    "sky": 16,
    "ego_mask": 31
  },
  "pixel_conventions": {
    "sky_mask": "255=sky, 0=non-sky",
    "ego_mask": "255=ego/self vehicle, 0=other",
    "semantic_edge_mask": "255=edge/filter candidate, 0=non-edge",
    "worldmirror_keep_mask": "255=keep, 0=filter out"
  },
  "outputs": {
    "sky_mask_dir": "masks/sky_mask",
    "ego_mask_dir": "masks/ego_mask",
    "semantic_edge_mask_dir": "masks/semantic_edge_mask",
    "worldmirror_keep_mask_dir": "masks/worldmirror_keep_mask"
  }
}
```

## 7. 推荐实施顺序

1. 先做 `seg_000` 的 dry-run：校验 `timestamps.txt`、`images/`、`camera_params.json`、`../source_calib.json` 数量和 ID 一致。
2. 跑 1 到 4 张 smoke test，保存 `image_segment/semantic`、`image_segment/undistorted` 和 `color_viz`。
3. 检查 `mask_stats.jsonl` 中是否出现 `16` 和 `31`；如果某段没有天空或没有 ego 区域，不应强行报错。
4. 从语义 PNG 批量生成 `masks/sky_mask`、`masks/ego_mask`、`masks/semantic_edge_mask`、`masks/worldmirror_keep_mask`。
5. 抽查图像尺寸：所有 mask 必须与对应去畸变图像和语义图尺寸一致。
6. 全量跑 `seg_000` 到 `seg_007`。
7. 如果后续要让外部 mask 真正参与 WorldMirror 推理，再扩展 `hyworld2/worldrecon/pipeline.py`，增加类似 `--external_sky_mask_dir` 或 `--external_keep_mask_dir` 的参数。

## 8. 验收项

每个 segment 至少应满足：

1. `image_segment/semantic/*.png` 数量等于 `timestamps.txt` 行数。
2. `image_segment/undistorted/*.jpeg` 数量等于 `timestamps.txt` 行数。
3. `masks/sky_mask/*.png`、`masks/ego_mask/*.png`、`masks/semantic_edge_mask/*.png`、`masks/worldmirror_keep_mask/*.png` 数量等于 `timestamps.txt` 行数。
4. `class_labels.json` 中包含 `"16": "sky"` 和 `"31": "ego_mask"`。
5. `mask_stats.jsonl` 每行有 `stem`、`height`、`width`、`labels`。
6. 所有二值 mask 只包含像素值 `0` 和 `255`。
7. `worldmirror_keep_mask` 的 `0` 区域至少覆盖 sky、ego 和 semantic edge；如果关闭 semantic edge，则需在 manifest 中记录。

## 9. 后续代码改造建议

建议新增一个面向 segment 的脚本，例如：

```text
scripts/generate_segment_masks.py
```

或放在 seg_mask 工程中：

```text
/home/haiming/admap_ws/seg_mask/mask2former/pipelines/infer_segment_masks.py
```

它应直接接受：

```bash
python infer_segment_masks.py \
  --segment-dir /home/haiming/open_source/world_models/HY-World-2.0/dataset/tracks/644470719/mid_center_top_wide/seg_000 \
  --source-calib /home/haiming/open_source/world_models/HY-World-2.0/dataset/tracks/644470719/mid_center_top_wide/source_calib.json \
  --model-path /home/haiming/admap_ws/seg_mask/mask2former/weights/mask2former_r50_cadata_1920x1080_90000.pt \
  --save-color-viz \
  --overwrite
```

批量入口再遍历：

```text
dataset/tracks/644470719/mid_center_top_wide/seg_*/
```

这样可以复用 Mask2Former V1 的模型推理代码，同时保持当前 mono WorldMirror dataset 的 per-segment 目录组织。
