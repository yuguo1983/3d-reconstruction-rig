#!/bin/bash
# dense_recon.sh — 稠密重建 + Poisson 网格
#
# 依赖: COLMAP 4.1+ (无 CUDA)
# 前置: 已跑完 §3.1-3.6 (sparse_test/0/ 下有 cameras.txt + images.txt + points3D.txt)
#
# 用法:
#   bash dense_recon.sh
#   bash dense_recon.sh --sparse ./sparse_test/0 --images ./colmap_images --out ./dense
#
# 输出:
#   dense/images/        去畸变后的图
#   dense/stereo/depth_maps/*.geometric.bin   每张图的稠密深度图
#   dense/stereo/normal_maps/*.geometric.bin  每张图法向图
#   dense/fused.ply                          稠密点云 (几百万点)
#   dense/meshed-poisson.ply                 Poisson 网格
#
# 耗时 (CPU):
#   - patch_match_stereo 是大头, 6 路 × 30 帧 1080p 大概 30-60 分钟
#   - stereo_fusion / poisson_mesher 几分钟

set -e

ROOT="E:/3dremodule"
SPARSE="$ROOT/sparse_test/0"
IMAGES="$ROOT/colmap_images"
OUT="$ROOT/dense"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sparse)   SPARSE="$2"; shift 2 ;;
        --images)   IMAGES="$2"; shift 2 ;;
        --out)      OUT="$2"; shift 2 ;;
        *)          echo "未知参数: $1"; exit 1 ;;
    esac
done

# 检查 colmap
if ! command -v colmap &> /dev/null; then
    echo "[错误] colmap 不在 PATH, 请先 conda install -c conda-forge colmap"
    echo "      或把 tools/colmap/bin 加到 PATH"
    exit 1
fi

if [[ ! -f "$SPARSE/cameras.txt" ]]; then
    echo "[错误] 找不到 $SPARSE/cameras.txt, 先跑完 §3.6 稀疏重建"
    exit 1
fi

echo "===== 1. image_undistorter ====="
echo "  输入 sparse: $SPARSE"
echo "  输入 images: $IMAGES"
echo "  输出:        $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
colmap image_undistorter \
    --image_path "$IMAGES" \
    --input_path "$SPARSE" \
    --output_path "$OUT" \
    --output_type COLMAP \
    --max_image_size 2000

echo
echo "===== 2. patch_match_stereo (CPU, 慢!) ====="
colmap patch_match_stereo \
    --workspace_path "$OUT" \
    --workspace_format COLMAP \
    --PatchMatchStereo.gpu_index -1 \
    --PatchMatchStereo.num_threads 4 \
    --PatchMatchStereo.max_image_size 2000

echo
echo "===== 3. stereo_fusion ====="
colmap stereo_fusion \
    --workspace_path "$OUT" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$OUT/fused.ply"

echo
echo "===== 4. poisson_mesher ====="
colmap poisson_mesher \
    --input_path "$OUT/fused.ply" \
    --output_path "$OUT/meshed-poisson.ply" \
    --PoissonMeshing.trim 5

echo
echo "===== 完成 ====="
echo "  稠密点云: $OUT/fused.ply"
echo "  Poisson 网格: $OUT/meshed-poisson.ply"
echo
echo "查看:  MeshLab / CloudCompare 打开 .ply"
echo "下一步: python texturize.py 把网格贴上颜色 → .obj"
