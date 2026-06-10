#!/bin/bash
# run_pipeline.sh — 端到端测试: 合成数据 → COLMAP rig pipeline → 稠密 + 网格 + 纹理
#
# 不需要真实硬件, 一键跑完整个流程并打印结果
# 在 git-bash / WSL 下可直接执行

set -e

ROOT="E:/3dremodule"
SYNTH="$ROOT/synthetic"
COLMAP_IMG="$ROOT/colmap_images"
RIG_CONFIG="$ROOT/rig_config.json"
DB="$ROOT/sparse_test/database.db"
SPARSE="$ROOT/sparse_test"
DENSE="$ROOT/dense"

# 检查 colmap
if ! command -v colmap &> /dev/null; then
    echo "[错误] colmap 不在 PATH, 请先 conda install -c conda-forge colmap"
    exit 1
fi

echo "===== 1. 生成合成数据 (30 帧 × 6 相机) ====="
python "$ROOT/make_synthetic_rig.py" --out "$SYNTH" --frames 30

echo
echo "===== 2. 重组目录 ====="
rm -rf "$COLMAP_IMG"
python "$ROOT/reorganize_captures.py" --src "$SYNTH" --dst "$COLMAP_IMG"

echo
echo "===== 3. 生成 rig_config.json ====="
python "$ROOT/rig_to_colmap_config.py" --calib "$SYNTH/rig_calib.json" --out "$RIG_CONFIG"

echo
echo "===== 4. COLMAP feature_extractor ====="
rm -rf "$SPARSE"; mkdir -p "$SPARSE"
colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$COLMAP_IMG" \
    --ImageReader.single_camera_per_folder 1 \
    --FeatureExtraction.use_gpu 0 \
    --FeatureExtraction.num_threads 4

echo
echo "===== 5. COLMAP rig_configurator ====="
colmap rig_configurator \
    --database_path "$DB" \
    --rig_config_path "$RIG_CONFIG"

echo
echo "===== 6. COLMAP exhaustive_matcher (with rig verification) ====="
colmap exhaustive_matcher \
    --database_path "$DB" \
    --FeatureMatching.use_gpu 0 \
    --FeatureMatching.rig_verification 1 \
    --FeatureMatching.num_threads 4

echo
echo "===== 7. COLMAP mapper (稀疏 SfM) ====="
colmap mapper \
    --database_path "$DB" \
    --image_path "$COLMAP_IMG" \
    --output_path "$SPARSE" \
    --Mapper.ba_use_gpu 0 \
    --Mapper.num_threads 4

# 检查 mapper 是否真的输出了
if [[ ! -f "$SPARSE/0/cameras.txt" ]]; then
    echo "[警告] mapper 没输出 (合成数据纹理重复时常见), 跳过稠密步骤"
    echo "查看 sparse: ls $SPARSE/"
    exit 0
fi

echo
echo "===== 8. 稠密重建 (image_undistorter + patch_match_stereo + stereo_fusion) ====="
bash "$ROOT/dense_recon.sh" --sparse "$SPARSE/0" --images "$COLMAP_IMG" --out "$DENSE"

echo
echo "===== 9. 网格 + 纹理 (poisson_mesher + texturize.py) ====="
python "$ROOT/texturize.py" --mesh "$DENSE/meshed-poisson.ply" --sparse "$DENSE"

echo
echo "===== 完成 ====="
echo "  合成数据:        $SYNTH"
echo "  重组目录:        $COLMAP_IMG"
echo "  rig_config:      $RIG_CONFIG"
echo "  COLMAP database: $DB"
echo "  稀疏重建:        $SPARSE/0/"
echo "  稠密点云:        $DENSE/fused.ply"
echo "  Poisson 网格:    $DENSE/meshed-poisson.ply"
echo "  纹理模型:        $DENSE/model.obj"
echo
echo "查看:"
echo "  MeshLab / CloudCompare:  $DENSE/fused.ply  或  $DENSE/model.obj"
echo "  ls $SPARSE/0/"
echo "  ls $DENSE/"
