"""
validate_rig_config.py — 跑 colmap rig_configurator 之前先验证 rig_config.json

模仿 COLMAP src/colmap/scene/rig.cc:ReadRigConfig() 的所有 THROW_CHECK,
提前在 Python 端发现错误, 免得到 COLMAP 里再炸。

用法:
    python validate_rig_config.py --config rig_config.json
    python validate_rig_config.py --config rig_config.json --image-dir ./colmap_images
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# 与 COLMAP src/colmap/sensor/models.h 一致
# (focal_params, principal_point_params, extra_distortion_params)
CAMERA_MODELS: dict[str, tuple[int, int, int]] = {
    "SIMPLE_PINHOLE": (1, 2, 0),
    "PINHOLE": (2, 2, 0),
    "SIMPLE_RADIAL": (1, 2, 1),
    "RADIAL": (1, 2, 2),
    "OPENCV": (2, 2, 4),
    "OPENCV_FISHEYE": (2, 2, 4),
    "FULL_OPENCV": (2, 2, 8),
    "FOV": (1, 2, 2),
    "SIMPLE_RADIAL_FISHEYE": (1, 2, 1),
    "RADIAL_FISHEYE": (1, 2, 2),
    "THIN_PRISM_FISHEYE": (2, 2, 4),
    "RADTAN_THIN_PRISM_FISHEYE": (2, 2, 8),
    "SIMPLE_DIVISION": (1, 2, 0),
    "DIVISION": (2, 2, 0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="可选, 校验 image_prefix 对应的子目录是否存在")
    args = ap.parse_args()

    errors = 0
    warnings = 0

    def fail(msg: str) -> int:
        print(f"  X {msg}")
        return 1

    def warn(msg: str) -> None:
        nonlocal warnings
        print(f"  ! {msg}")
        warnings += 1

    print(f"=== 校验 {args.config} ===")
    try:
        rigs = json.loads(args.config.read_text())
    except json.JSONDecodeError as e:
        return fail(f"JSON 解析失败: {e}")

    if not isinstance(rigs, list) or len(rigs) == 0:
        errors += fail("顶层必须是非空数组 [...]")
        return errors
    print(f"顶层 rig 数: {len(rigs)}")

    for rig_idx, rig in enumerate(rigs):
        print(f"\n[rig #{rig_idx}]")
        if "cameras" not in rig or not rig["cameras"]:
            errors += fail("  缺 'cameras' 或为空")
            continue
        cams = rig["cameras"]
        print(f"  相机数: {len(cams)}")

        ref_count = 0
        prefixes: set[str] = set()

        for ci, cam in enumerate(cams):
            tag = f"  cam{ci}:"

            if "image_prefix" not in cam:
                errors += fail(f"{tag} 缺 'image_prefix'")
                continue
            prefix = cam["image_prefix"]
            if not prefix.endswith("/"):
                warn(f"{tag} image_prefix='{prefix}' 建议以 '/' 结尾")
            if prefix in prefixes:
                errors += fail(f"{tag} 重复的 image_prefix '{prefix}'")
            prefixes.add(prefix)
            print(f"{tag} image_prefix='{prefix}'")

            is_ref = bool(cam.get("ref_sensor", False))
            if is_ref:
                ref_count += 1
                if "cam_from_rig_rotation" in cam or "cam_from_rig_translation" in cam:
                    errors += fail(f"{tag} ref_sensor 不能有 cam_from_rig_*")
                print(f"{tag} role: REF")
            else:
                for k in ("cam_from_rig_rotation", "cam_from_rig_translation"):
                    if k not in cam:
                        errors += fail(f"{tag} 非 ref 缺 '{k}'")
                if "cam_from_rig_rotation" in cam:
                    q = cam["cam_from_rig_rotation"]
                    if len(q) != 4:
                        errors += fail(f"{tag} quat 长度 {len(q)} != 4")
                    else:
                        n = sum(x*x for x in q) ** 0.5
                        if abs(n - 1.0) > 0.01:
                            warn(f"{tag} quat 模长 {n:.4f} != 1 (COLMAP 会 normalize)")
                if "cam_from_rig_translation" in cam:
                    t = cam["cam_from_rig_translation"]
                    if len(t) != 3:
                        errors += fail(f"{tag} translation 长度 {len(t)} != 3")

            has_model = "camera_model_name" in cam
            has_params = "camera_params" in cam
            if has_model != has_params:
                errors += fail(f"{tag} camera_model_name/camera_params 必须同时给")
            elif has_model:
                model = cam["camera_model_name"]
                if model not in CAMERA_MODELS:
                    errors += fail(f"{tag} 未知 camera_model_name '{model}' "
                                   f"(支持: {', '.join(sorted(CAMERA_MODELS.keys()))})")
                else:
                    foc, pp, ext = CAMERA_MODELS[model]
                    expected = foc + pp + ext
                    actual = len(cam["camera_params"])
                    if actual != expected:
                        errors += fail(
                            f"{tag} '{model}' 期望 {expected} params (focal={foc} pp={pp} extra={ext}), 实际 {actual}")
                    else:
                        print(f"{tag} camera_model: {model} ({actual} params)")

        if ref_count == 0:
            errors += fail("  rig 至少要有 1 个 ref_sensor")
        elif ref_count > 1:
            errors += fail(f"  rig 只能有 1 个 ref_sensor, 实际 {ref_count}")
        else:
            print("  OK 恰好 1 个 ref_sensor")

    if args.image_dir and args.image_dir.exists():
        print(f"\n=== image dir 校验: {args.image_dir} ===")
        for rig in rigs:
            for cam in rig.get("cameras", []):
                prefix = cam.get("image_prefix", "").rstrip("/")
                if not prefix:
                    continue
                sub = args.image_dir / prefix
                if not sub.exists():
                    errors += fail(f"  '{prefix}/' 在 {args.image_dir} 下不存在")
                else:
                    n = sum(1 for x in sub.iterdir() if x.is_file())
                    print(f"  OK {prefix}/ ({n} 文件)")
                    if n == 0:
                        warn(f"  '{prefix}/' 存在但为空, 该相机会被建为 trivial rig")

    print(f"\n=== 总结 ===")
    print(f"  错误: {errors}  警告: {warnings}")
    if errors == 0:
        print("OK rig_config.json 可以喂给 colmap rig_configurator")
        return 0
    print("X 有错误, 修完再跑 COLMAP")
    return 1


if __name__ == "__main__":
    sys.exit(main())
