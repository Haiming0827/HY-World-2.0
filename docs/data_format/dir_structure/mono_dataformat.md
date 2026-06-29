# HY-World 单目前馈式 3DGS 推理数据格式

版本：V0.1

本文面向 HY-World 2.0 / WorldMirror 2.0 的单目前视数据接入，基于以下单程样例分析：

```text
/media/haiming/data2/2026_test/0521_ramp/7869148/pipeline/public/MX11-2S/HXMQBC7PZ4CT25SL1/2026-03-21/110247/110557/644470719
```

目标相机：

```text
camera/mid_center_top_wide
```

SFM 优化位姿：

```text
deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json
```

## 1. 输入数据源

本样例中用于单目 WorldMirror 推理数据集制作的关键文件如下：

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 原始标定 | `calib.json` | 全车传感器标定，含 `mid_center_top_wide` 内外参 |
| 相机索引 | `camera.json` | 按相机名组织的文件列表和标定副本 |
| 前视图像 | `camera/mid_center_top_wide/<timestamp_ns>.jpeg` | 3840x2160 RGB JPEG，文件名为纳秒时间戳 |
| SFM 位姿 | `deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json` | 191 个优化关键帧位姿，时间戳可与图像文件名 exact match |
| 根目录位姿副本 | `optimized_keyframe_poses.json` | 也是 191 条，但 `optimized_source/track_id/quat` 与 deep-learning 结果略有差异；本文以 deep-learning 结果为准 |

## 2. 图像目录格式

WorldMirror CLI 原生支持直接读取一个图片目录：

```text
<input_images>/
├── 1774062167714436374.jpeg
├── 1774062167947755496.jpeg
├── 1774062168181095327.jpeg
└── ...
```

要求：

- 图片扩展名支持 `.jpeg`、`.jpg`、`.png`、`.webp`。
- 文件名 stem 建议使用时间戳，例如 `1774062167714436374`。
- 同一次推理输入内，所有图像应具有相同宽高比例；WorldMirror 会按 `target_size` resize 并 center crop 到 patch size 的整数倍。
- 如要使用相机先验，`camera_params.json` 中的 `camera_id` 应与图像文件 stem 对齐，或使用从 0 开始的连续整数索引。

样例统计：

| 项 | 值 |
| --- | --- |
| 图像目录 | `camera/mid_center_top_wide` |
| 图像数量 | 5698 |
| 图像分辨率 | 3840x2160 |
| 图像格式 | JPEG / RGB |
| 图像时间戳首帧 | `1774062167081105386` |
| 图像时间戳末帧 | `1774062356981097387` |
| 图像间隔 | median 33.334339 ms，约 30 FPS |

## 3. `mid_center_top_wide` 标定参数

在 `calib.json` 的 `cameras` 列表中，目标相机条目满足：

```json
{
  "name": {
    "label": "SENSOR_LABEL_MID_CENTER_TOP_WIDE",
    "alias": "sunyu_imx728"
  }
}
```

基础参数：

| 字段 | 值 |
| --- | --- |
| label | `SENSOR_LABEL_MID_CENTER_TOP_WIDE` |
| alias | `sunyu_imx728` |
| width | `3840` |
| height | `2160` |
| fov | `120` |
| camera model | `CAMERA_MODEL_KANNEL_BRAND` |
| distortion model | `kb_model` |
| line_delay_ns | `0` |
| status | `CALIB_STATUS_ONLINE_SELF_CALIBRATED` |

内参矩阵可由 `kb_model` 中的 `fu/fv/pu/pv` 组成：

```text
K =
[[1904.99609375,    0.0,        1917.16064453125],
 [   0.0,        1903.80493164, 1076.62536621094],
 [   0.0,           0.0,           1.0]]
```

畸变参数：

```text
kb_distortion = [
  -0.029459979385137558,
  -0.005299502518028021,
  -0.0012424745364114642,
   0.0012543649645522237
]
```

标定中的 `sensor_to_vehicle`：

```json
{
  "quaternion_w": 0.7000109708349931,
  "quaternion_x": -0.7141304467010249,
  "quaternion_y": -0.00025297599813983235,
  "quaternion_z": 0.0015105655859465779,
  "translation_x_m": 0.0020393036122394095,
  "translation_y_m": 1.854559047705093,
  "translation_z_m": 1.4109925226500386
}
```

按 `wxyz` 四元数转换得到的 4x4 矩阵为：

```text
T_sensor_to_vehicle =
[[ 9.99995308e-01, -1.75350924e-03, -2.51165370e-03,  2.03930361e-03],
 [ 2.47614069e-03, -1.99691534e-02,  9.99797530e-01,  1.85455905e+00],
 [-1.80330981e-03, -9.99799059e-01, -1.99647178e-02,  1.41099252e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
```

标定中的 `sensor_to_vehicle_eol`：

```json
{
  "quaternion_w": 0.700973551062791,
  "quaternion_x": -0.7131866060337481,
  "quaternion_y": 0.0008918387627634325,
  "quaternion_z": 0.0003876958929940931,
  "translation_x_m": 0.0020393036122394095,
  "translation_y_m": 1.854559047705093,
  "translation_z_m": 1.4109925226500386
}
```

转换矩阵：

```text
T_sensor_to_vehicle_eol =
[[ 9.99998109e-01, -1.81562405e-03,  6.97311733e-04,  2.03930361e-03],
 [-7.28565787e-04, -1.72705707e-02,  9.99850587e-01,  1.85455905e+00],
 [-1.80330981e-03, -9.99849204e-01, -1.72718608e-02,  1.41099252e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
```

另有两个位置字段：

```json
"sensor_position_to_vcs": {
  "euler_x_rad": 0.0018158958884099989,
  "euler_y_rad": 0.01727142933971048,
  "euler_z_rad": -0.0007286745314063404,
  "position_x_m": 1.854559047705093,
  "position_y_m": -0.0020393036122394095,
  "position_z_m": 1.4109925226500386
}
```

```json
"local_coordinate_to_vcs": {
  "euler_x_rad": 0.0,
  "euler_y_rad": 0.0,
  "euler_z_rad": 0.0,
  "position_x_m": 3.86,
  "position_y_m": 0.0,
  "position_z_m": 0.0
}
```

注意：WorldMirror `prior_cam_path` 需要的是 OpenCV convention 的 camera-to-world `c2w` 和 pinhole 内参矩阵。上面的 `sensor_to_vehicle` 是传感器到车体系的外参，需要与每帧车辆或相机位姿组合，并处理坐标系约定后，才能生成 WorldMirror 可用的 `extrinsics.matrix`。

## 4. SFM 位姿格式与时间戳匹配

`deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json` 是数组，样例长度为 191。单条记录格式：

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

时间戳对齐统计：

| 项 | 值 |
| --- | ---: |
| 前视图像数量 | 5698 |
| SFM 位姿数量 | 191 |
| exact match 数量 | 191 |
| 未匹配位姿数量 | 0 |
| 最近邻时间戳误差 | min/median/mean/max 均为 0 ns |
| 位姿首帧时间戳 | `1774062167714436374` |
| 位姿末帧时间戳 | `1774062356614437178` |
| 位姿覆盖图像索引范围 | 第 19 张到第 5686 张 |
| 图像首帧到位姿首帧偏移 | 633.330988 ms |
| 位姿末帧到图像末帧剩余 | 366.660209 ms |

结论：SFM 优化位姿时间戳与 `mid_center_top_wide` 图像文件名可以完全对上。可直接用 `pose.timestamp` 或 `pose.name` 匹配同名 JPEG。

位姿关键帧并不是连续 30 FPS，而是关键帧抽样。常见步长约 6-15 张图，存在若干大间隔：

| 位姿索引跳变 | 间隔 |
| --- | ---: |
| 41 -> 42 | 12.666656965 s |
| 54 -> 55 | 15.933325581 s |
| 90 -> 91 | 20.333327470 s |
| 157 -> 158 | 44.700002705 s |

这些大间隔意味着：如果直接把 191 个关键帧作为一次 WorldMirror 推理输入，时间覆盖跨度很大，局部连续性不均匀。实际制作数据集时应按连续片段切分，避免把长时间断裂的关键帧放进同一个小 batch。

## 5. 重要坐标系风险

`optimized_keyframe_poses.json` 的 `pose.trans` 数值范围为：

```text
min = [120.14875771995887, 30.304571737376648, 11.829010959798545]
max = [120.15011920034881, 30.305922915097817, 17.80796624721914]
```

这个数值非常接近 `longitude / latitude / height`，而不像米制局部坐标。根目录 `loc.json` 的首条记录也含：

```json
{
  "latitude": 30.30592308041688,
  "longitude": 120.14875688727824,
  "height": 11.917257041635455
}
```

因此，不应直接把 `pose.trans` 当作 WorldMirror `camera_params.json` 的米制 c2w 平移。需要先确认 `optimized_keyframe_poses.json` 的语义：

1. 如果 `trans` 是经纬高，则必须转换到局部 ENU/UTM，并选择一致原点。
2. 如果 `quat` 是车体在地理坐标系下的姿态，则需要组合 `T_sensor_to_vehicle` 或其逆，并转换到相机 OpenCV 坐标。
3. 如果 `pose` 已经是某种相机坐标或 SFM 坐标，则需要找到该坐标系定义和尺度单位，再转 WorldMirror 所需 c2w。

对于 HY-World / WorldMirror 的 `prior_cam_path`，推荐最终落盘为：

```json
{
  "num_cameras": 3,
  "extrinsics": [
    {
      "camera_id": "1774062167714436374",
      "matrix": [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]
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

其中 `extrinsics.matrix` 必须是 OpenCV convention 的 camera-to-world 4x4 矩阵；`intrinsics.matrix` 应对应原始图像分辨率，WorldMirror 会在推理时自动根据 resize + center crop 调整内参。

## 6. 建议的数据集目录格式

建议把从单程目录抽出的 HY-World 单目样本组织为：

```text
<mono_dataset_root>/
└── <track_id>_<camera_name>/
    ├── images/
    │   ├── 1774062167714436374.jpeg
    │   ├── 1774062167947755496.jpeg
    │   └── ...
    ├── camera_params.json
    ├── source_calib.json
    ├── source_poses.json
    ├── timestamps.txt
    ├── segments.json
    └── meta.json
```

字段说明：

| 文件 | 必需 | 说明 |
| --- | --- | --- |
| `images/` | 是 | WorldMirror 输入图片目录。建议只放待推理帧，不放全量 5698 帧 |
| `camera_params.json` | 可选但推荐 | WorldMirror prior camera 文件，含 c2w 和 K |
| `source_calib.json` | 推荐 | 抽取出的 `mid_center_top_wide` 原始标定，便于追溯 |
| `source_poses.json` | 推荐 | 与图片 exact match 的原始 SFM 位姿记录 |
| `timestamps.txt` | 推荐 | 每行一个时间戳，顺序与图片排序一致 |
| `segments.json` | 推荐 | 连续片段划分，避免大时间断裂 |
| `meta.json` | 推荐 | 数据来源、相机名、轨迹 ID、版本、处理参数 |

`timestamps.txt`：

```text
1774062167714436374
1774062167947755496
1774062168181095327
```

`segments.json`：

```json
{
  "segments": [
    {
      "segment_id": "seg_000",
      "start_timestamp": 1774062167714436374,
      "end_timestamp": 1774062178281101073,
      "frame_count": 41,
      "reason": "continuous_keyframes"
    }
  ]
}
```

`meta.json`：

```json
{
  "source_single_track_dir": "/media/haiming/data2/2026_test/0521_ramp/7869148/pipeline/public/MX11-2S/HXMQBC7PZ4CT25SL1/2026-03-21/110247/110557/644470719",
  "camera_name": "mid_center_top_wide",
  "camera_label": "SENSOR_LABEL_MID_CENTER_TOP_WIDE",
  "track_id": "644470719",
  "image_width": 3840,
  "image_height": 2160,
  "image_count_raw": 5698,
  "pose_count": 191,
  "timestamp_unit": "ns",
  "pose_source": "deep-learning/vision-map/20250411160204/results/optimized_keyframe_poses.json",
  "pose_timestamp_exact_match": true,
  "pose_coordinate_status": "must_convert_or_confirm_before_using_as_worldmirror_prior"
}
```

## 7. WorldMirror 直接推理输入

如果只使用图像，不注入外参/内参先验，最小输入就是一个图片目录：

```bash
python -m hyworld2.worldrecon.pipeline \
  --input_path <mono_dataset_root>/<track_id>_<camera_name>/images \
  --strict_output_path <output_dir> \
  --target_size 392 \
  --enable_bf16 \
  --no_interactive
```

如果生成了可用的 `camera_params.json`：

```bash
python -m hyworld2.worldrecon.pipeline \
  --input_path <mono_dataset_root>/<track_id>_<camera_name>/images \
  --prior_cam_path <mono_dataset_root>/<track_id>_<camera_name>/camera_params.json \
  --strict_output_path <output_dir> \
  --target_size 392 \
  --enable_bf16 \
  --no_interactive
```

注意：

- `camera_params.json` 可以只提供 intrinsics，也可以同时提供 extrinsics 和 intrinsics。
- `camera_id` 可以使用图片文件 stem，例如 `"1774062167714436374"`。
- 如果提供 extrinsics，必须保证它是 WorldMirror 期望的 OpenCV c2w，而不是未经转换的经纬高或车体位姿。
