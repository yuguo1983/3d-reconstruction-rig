# 3dremodule — 6 路固定相机 COLMAP 重建流水线

针对**静态小物件**的离线高质量 3D 重建方案：6 个 UVC 相机以**软件时间戳**同步
抓拍,经过 ChArUco 联合标定后,喂给 COLMAP 4.x 的 **rig pipeline** 做稀疏重建。

> 不依赖硬件触发线,不需要 NVIDIA GPU(用 COLMAP CPU SIFT + 单线程 BA)。

---

## 1. 项目结构

```
3dremodule/
├── README.md                    ← 你正在看的
├── rig_capture.py               1. 多路 UVC 同步抓帧
├── reorganize_captures.py       2. 目录重组: gid_cam.jpg → cam/gid.jpg
├── calibrate_rig.py             3. 6 路 ChArUco 联合标定
├── rig_to_colmap_config.py      4. 标定 → COLMAP rig_config.json
├── validate_rig_config.py       5. rig_config.json 自检(模仿 COLMAP THROW_CHECK)
├── compare_calib.py             6a. 把标定结果跟 ground truth 对比
├── compare_reconstruction.py    6b. 把 COLMAP 重建结果跟 GT 对比
├── make_synthetic_rig.py        A. 合成 6 路数据(无硬件也能跑通整条流水线)
├── run_pipeline.sh              B. 合成数据 → COLMAP 端到端一键脚本
├── rig_ui.py                    C. PyQt5 桌面端: 2×3 实时预览 + 一键抓帧
│
├── rig.json                     相机索引/分辨率/曝光/对焦的配置
├── tools/colmap/                COLMAP 4.1.0(无 CUDA)
│   └── bin/colmap.exe
├── captures/                    rig_capture.py 的输出(按 session 子目录)
├── colmap_images/               reorganize 后的 COLMAP 输入目录
├── calib/                       calibrate_rig.py 的输出
├── synthetic/                   make_synthetic_rig.py 的输出
└── sparse_test/                 COLMAP mapper 的输出
```

---

## 2. 硬件要求

| 项目 | 要求 |
|------|------|
| 相机 | 6× UVC 摄像头(同型号更佳),索引固定 |
| 分辨率 | 1920×1080(代码里写死,改 `rig_capture.py` 的 `CamConfig` 即可) |
| 帧率 | 30 FPS(由 USB 带宽决定:6×1080p30 MJPEG ≈ 1.8 Gbps,USB3.0 必需) |
| 触发 | 无硬件触发 — 用 `time.monotonic_ns()` 软件时间戳 + `sync_window_ms` 聚类 |
| 镜头 | 6 个**完全相同焦距**的定焦镜头(M12/CS 接口),避免 SIFT 误匹配 |
| 标定板 | 1 块 ChArUco 板(9×7 方格, 20mm square, 14mm marker) |
| 算力 | 无 GPU;COLMAP 走 SIFT CPU + OpenMP,8 核约 5-10 分钟/30 帧 |

---

## 3. 端到端流水线

按顺序跑下面 7 步,每一步都打印验证信息,**任何一步失败都先解决再走下一步**。

### 3.1 相机抓帧

**图形界面(推荐)**: `python rig_ui.py` 打开 PyQt5 桌面端, 6 路实时预览 + 一键抓帧, 无需记命令.

**命令行**:
```bash
# 采集 30 组(每组 6 张)ChArUco 板多角度图
python rig_capture.py --config rig.json --count 30 --output ./captures
```

输出: `./captures/<YYYYMMDD_HHMMSS>/00000_cam0_north.jpg ... 00029_cam5_northwest.jpg`

**关键参数**(`rig.json`):
- `sync_window_ms: 10` — 6 路时间戳落在这个窗内才算"同步",> 10ms 重新聚类
- `exposure: -7.0, focus: 80` — 固定曝光和对焦,避免自动模式帧间跳变

> **第一次跑建议先 `--count 1`**,确认 6 路都能正常打开、图像方向正确、组内时间跨度 < 10ms。

### 3.2 目录重组

```bash
# 把 00000_cam0_north.jpg 拆成 colmap_images/cam0_north/00000.jpg
python reorganize_captures.py --src ./captures/<session> --dst ./colmap_images
```

输出结构:
```
colmap_images/
  cam0_north/      00000.jpg  00001.jpg  ...
  cam1_northeast/  00000.jpg  00001.jpg  ...
  ...
  cam5_northwest/  00000.jpg  00001.jpg  ...
```

COLMAP rig 约定:每个子目录一台相机,同名 `.jpg` 视为同一帧。

### 3.3 6 路 ChArUco 联合标定

```bash
python calibrate_rig.py --captures ./captures --out ./calib
```

输出(`./calib/`):
- `rig_calib.json` — 内参 + 外参(R,T 是 **cam_from_ref**) + 完整标定元数据
- `cameras.txt` — COLMAP 格式的相机内参(可直接喂 COLMAP)
- `images.txt` — 第一帧 6 张图的 COLMAP 位姿(便于目视检查)
- `rig_layout.png` — 6 台相机的 3D 俯视图(世界系 = 参考相机)

**算法**:
- 内参:每台相机独立 `cv2.calibrateCamera` + RMS 重投影误差自检
- 外参:每对 `cv2.stereoCalibrate(..., CALIB_FIX_INTRINSIC)`
  - 关键:按 **ChArUco ID** 对齐共同角点(不是按 frame index),避免单帧漏检造成数组长度不匹配

**已知精度问题**:`cv2.stereoCalibrate` 在小基线(基线 < 0.5m) + 小物体(场景 < 0.2m) 下
平移尺度可能偏 5-10×(OpenCV 的经典坑)。**真实场景(基线 ~0.5m, 物体 ~0.3m)问题不大**;
合成数据因为标定板只占视野中央一小块,问题更明显。
补救方案:多角度大姿态采集(板要经常倾斜)、加 `CALIB_USE_EXTRINSIC_GUESS` 提供先验。

### 3.4 标定 → COLMAP rig_config.json

```bash
python rig_to_colmap_config.py --calib ./calib/rig_calib.json --out ./rig_config.json
```

产出符合 COLMAP `src/colmap/scene/rig.h:42-120` 的 schema:

```json
[{
  "cameras": [
    {
      "image_prefix": "cam0_north/",
      "ref_sensor": true,
      "camera_model_name": "OPENCV",
      "camera_params": [fx, fy, cx, cy, k1, k2, p1, p2]
    },
    {
      "image_prefix": "cam1_northeast/",
      "cam_from_rig_rotation": [w, x, y, z],
      "cam_from_rig_translation": [tx, ty, tz],
      "camera_model_name": "OPENCV",
      "camera_params": [fx, fy, cx, cy, k1, k2, p1, p2]
    },
    ...
  ]
}]
```

约定:
- `R, T` 来自 `calibrate_rig.py`,**本身就是 `cam_from_ref`**,直接当 `cam_from_rig` 用
- 参考相机(`ref_sensor: true`)的 R/T 字段必须**不写**,COLMAP 会 THROW_CHECK 失败

### 3.5 rig_config.json 自检

```bash
python validate_rig_config.py --config ./rig_config.json --image-dir ./colmap_images
```

这一脚本**模仿 COLMAP `src/colmap/scene/rig.cc:ReadRigConfig()` 的所有 THROW_CHECK**,
提前在 Python 端发现:
- 缺/重复的 `image_prefix`
- 非 ref 相机缺 `cam_from_rig_rotation/translation`
- 四元数模长异常(> 0.01 偏差)
- `camera_model_name` 拼错、`camera_params` 长度不对
- `image_prefix` 指向不存在的子目录

> 跑 COLMAP 之前**先**跑这个,可以省掉大量 "unrecognised option / Inconsistent cameras"
> 之类的无厘头错误。

### 3.6 COLMAP 重建

```bash
export PATH="$PWD/tools/colmap/bin:$PATH"
DB=./sparse_test/database.db
SPARSE=./sparse_test

mkdir -p $SPARSE
rm -f $DB

# 1. 提特征
colmap feature_extractor \
    --database_path $DB \
    --image_path ./colmap_images \
    --ImageReader.single_camera_per_folder 1 \
    --FeatureExtraction.use_gpu 0 \
    --FeatureExtraction.num_threads 4

# 2. 把 rig_config.json 写入数据库
colmap rig_configurator \
    --database_path $DB \
    --rig_config_path ./rig_config.json

# 3. 匹配(开启 rig 跨相机验证)
colmap exhaustive_matcher \
    --database_path $DB \
    --FeatureMatching.use_gpu 0 \
    --FeatureMatching.rig_verification 1 \
    --FeatureMatching.num_threads 4

# 4. 稀疏重建
colmap mapper \
    --database_path $DB \
    --image_path ./colmap_images \
    --output_path $SPARSE \
    --Mapper.ba_use_gpu 0 \
    --Mapper.num_threads 4
```

**关键 flag 说明**:

| Flag | 作用 | 必须? |
|------|------|-------|
| `--ImageReader.single_camera_per_folder 1` | 每个子目录一台相机(否则 rig 失效) | **必须** |
| `--FeatureMatching.rig_verification 1` | 跨相机匹配时利用刚体约束剔除外点 | **强烈建议** |
| `--FeatureExtraction.use_gpu 0` | 无 GPU,纯 CPU SIFT | 无 GPU 时必加 |
| `--Mapper.ba_use_gpu 0` | BA 走 CPU Ceres | 无 GPU 时必加 |

**输出**:
```
sparse_test/
  database.db
  0/
    cameras.txt
    images.txt
    points3D.txt
  1/  2/ ...   (如果有多个模型)
```

### 3.7 验证重建质量

```bash
# 跟 ground truth 比(仅合成数据可用)
python compare_reconstruction.py \
    --sparse ./sparse_test \
    --gt-calib ./synthetic/rig_calib.json
```

打印:
- 内参误差 Δfx/Δfy/Δcx/Δcy
- 外参误差 R 角度误差(度)、T 平移误差(mm 和 %)
- 已注册图像数(`0 / 180` = 全部没接上,需要回头查)

---

## 4. 合成数据一键跑通(无硬件也能验证)

```bash
# 在 git-bash / WSL 下
bash run_pipeline.sh
```

会自动跑完 1-7 步(标定直接用 GT 而非 `stereoCalibrate`,绕开小基线精度坑),
预期结果:
- 6 路 ChArUco 全部检出 48/48 角点
- `database.db` ≈ 30 MB
- `sparse_test/0/` 有非空 `images.txt`(合成数据纹理重复,SIFT 注册率会偏低)

**合成数据为什么能验证标定**:`make_synthetic_rig.py` 直接给出 ground truth
K + 6 个外参 + 渲染好的图,跳过 `cv2.stereoCalibrate` 这个最不稳定的环节,
专注于验证 COLMAP 端的 rig pipeline 接通。

---

## 5. 关键文件说明

| 文件 | 内容 | 何时读 |
|------|------|--------|
| `rig.json` | 6 台相机的索引/分辨率/曝光/对焦 | 跑 `rig_capture.py` 前必改 |
| `synthetic/rig_calib.json` | GT 标定(合成的 ground truth) | 评估标定精度时 |
| `calib/rig_calib.json` | `calibrate_rig.py` 的实际输出 | 喂给 3.4 |
| `rig_config.json` | COLMAP 格式的标定 | 喂给 `rig_configurator` |
| `colmap_images/<cam>/*.jpg` | 重组成按相机分类的图 | COLMAP `--image_path` |
| `sparse_test/0/images.txt` | 重建出的每张图位姿(qvec, tvec) | 评估重建质量 |

---

## 6. 已知问题 & 避坑

| 问题 | 触发条件 | 解决 |
|------|----------|------|
| `cv2.stereoCalibrate` 尺度偏 5-10× | 小基线(< 0.5m)+ 小物体 + 标定板只占视野中央 | 多角度大姿态、增大标定板相对视野、合成数据用 GT |
| `Inconsistent cameras` 错误 | `feature_extractor` 没加 `--ImageReader.single_camera_per_folder 1` | 加上即可 |
| `No images with matches` | 忘了跑 `exhaustive_matcher` | 必须先匹配再 mapper |
| `unrecognised option --SiftExtraction.use_gpu` | 旧版 COLMAP 文档误导 | 正确 flag 是 `--FeatureExtraction.use_gpu` |
| COLMAP 4.1 默认带 CUDA,Windows 上跑不动 | 没 NVIDIA 驱动 | 用 `tools/bin/colmap.exe`(已下无 CUDA 版本) |
| ChArUco 检测率低(每帧 < 8 角点) | 光照不均、对比度低、运动模糊 | 环形光 + 漫反射板 + 增大 `min_corners` 容差 |
| 合成图 SIFT 匹配率低(只有 5-10/30 帧注册) | 纹理重复 + 视角跳变 | 真实场景不会出现,合成数据是已知问题 |

---

## 7. 后续 TODO

- [ ] **真实硬件冒烟测试**:6 路 UVC 接入,跑通 §3 全流程,目视检视 `sparse_test/0/`
- [ ] **稠密重建**:`colmap patch_match_stereo` + `stereo_fusion` 生成稠密点云
- [ ] **网格重建**:`colmap poisson_mesher` 或直接 Open3D Poisson
- [ ] **纹理贴图**:用 mvs-texturing 把 RGB 贴到 mesh 上
- [ ] **改进 `calibrate_rig.py`**:加 `CALIB_USE_EXTRINSIC_GUESS`,先 PnP 给先验再 stereo refine
- [ ] **跨平台**:把 `dshow` backend 改成可切换(v4l2 / msmf / dshow)

---

## 8. 一行命令速查

```bash
# 全流程(合成)
bash run_pipeline.sh

# 全流程(真实硬件)
python rig_capture.py --config rig.json --count 30 --output ./captures
python reorganize_captures.py --src ./captures/<latest> --dst ./colmap_images
python calibrate_rig.py --captures ./captures --out ./calib
python rig_to_colmap_config.py --calib ./calib/rig_calib.json --out ./rig_config.json
python validate_rig_config.py --config ./rig_config.json --image-dir ./colmap_images
colmap feature_extractor --database_path ./sparse_test/database.db --image_path ./colmap_images --ImageReader.single_camera_per_folder 1 --FeatureExtraction.use_gpu 0 --FeatureExtraction.num_threads 4
colmap rig_configurator --database_path ./sparse_test/database.db --rig_config_path ./rig_config.json
colmap exhaustive_matcher --database_path ./sparse_test/database.db --FeatureMatching.use_gpu 0 --FeatureMatching.rig_verification 1 --FeatureMatching.num_threads 4
colmap mapper --database_path ./sparse_test/database.db --image_path ./colmap_images --output_path ./sparse_test --Mapper.ba_use_gpu 0 --Mapper.num_threads 4
python compare_reconstruction.py --sparse ./sparse_test --gt-calib ./synthetic/rig_calib.json
```
