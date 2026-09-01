"""数据集标注格式校验：越界框、损坏图像、异常标注。

面向万级图像：逐张流式检查，默认只打印摘要和前若干条问题，避免写出巨型报告。

示例::

    python dataset/validate_annotations.py --root data --max-report 30
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_CLASSES,
    dump_json,
    iter_image_files,
    label_path_for,
    parse_yolo_file,
    resolve_dirs,
    yolo_to_xyxy,
)

# 像素框相对原图面积过小 / 过大视为异常（可按现场再调）
MIN_BOX_AREA_RATIO = 1e-6
MAX_BOX_AREA_RATIO = 0.95
# 极端长宽比
MAX_ASPECT = 50.0
# 单图框数上限（电力巡检单帧通常远小于此）
MAX_BOXES_PER_IMAGE = 200
# 归一化坐标允许的轻微数值误差
NORM_EPS = 1e-3


def _imread_unicode(path: Path):
    """Windows 中文路径下 cv2.imread 可能失败，改用缓冲解码。"""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None, "空文件"
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None, "无法解码（文件损坏或格式不支持）"
    return img, None


def check_boxes(
    boxes: list[tuple[int, float, float, float, float]],
    width: int,
    height: int,
    num_classes: int,
) -> list[str]:
    """检查类别非法、归一化越界、像素越界、面积/长宽比异常、重复框。"""
    issues: list[str] = []
    if len(boxes) > MAX_BOXES_PER_IMAGE:
        issues.append(f"单图框数异常: {len(boxes)} > {MAX_BOXES_PER_IMAGE}")

    seen: set[tuple[int, str, str, str, str]] = set()

    for idx, (cls_id, xc, yc, w, h) in enumerate(boxes):
        prefix = f"框#{idx}"
        if cls_id < 0 or cls_id >= num_classes:
            issues.append(f"{prefix} 类别越界 class_id={cls_id}（有效 0..{num_classes - 1}）")
        for name, val in (("xc", xc), ("yc", yc), ("w", w), ("h", h)):
            if not np.isfinite(val):
                issues.append(f"{prefix} {name} 非有限数: {val}")
        if w <= 0 or h <= 0:
            issues.append(f"{prefix} 宽或高非正: w={w}, h={h}")
        # 中心点应在图内；宽高归一化应落在 (0,1]
        if not (-NORM_EPS <= xc <= 1.0 + NORM_EPS and -NORM_EPS <= yc <= 1.0 + NORM_EPS):
            issues.append(f"{prefix} 中心点越界: xc={xc:.4f}, yc={yc:.4f}")
        if w > 1.0 + NORM_EPS or h > 1.0 + NORM_EPS:
            issues.append(f"{prefix} 归一化宽高越界: w={w:.4f}, h={h:.4f}")

        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, width, height)
        if x1 < -1 or y1 < -1 or x2 > width + 1 or y2 > height + 1:
            issues.append(
                f"{prefix} 像素框越界: xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) 图像={width}x{height}"
            )

        area_ratio = (w * h)
        if 0 < area_ratio < MIN_BOX_AREA_RATIO:
            issues.append(f"{prefix} 框面积过小 ratio={area_ratio:.2e}")
        if area_ratio > MAX_BOX_AREA_RATIO:
            issues.append(f"{prefix} 框面积过大 ratio={area_ratio:.3f}")
        aspect = max(w / h, h / w) if w > 0 and h > 0 else 0.0
        if aspect > MAX_ASPECT:
            issues.append(f"{prefix} 长宽比异常 aspect={aspect:.1f}")

        key = (cls_id, f"{xc:.4f}", f"{yc:.4f}", f"{w:.4f}", f"{h:.4f}")
        if key in seen:
            issues.append(f"{prefix} 与已有框重复")
        seen.add(key)
    return issues


def validate_dataset(
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
    *,
    recursive: bool = True,
    progress_every: int = 500,
) -> dict:
    """流式校验。返回计数摘要 + 截断后的问题列表。"""
    stats: Counter[str] = Counter()
    samples: list[dict] = []

    n_img = 0
    for image_path in iter_image_files(images_dir, recursive=recursive):
        n_img += 1
        if progress_every and n_img % progress_every == 0:
            print(f"[validate] 已扫描 {n_img} 张 ...", flush=True)

        rec = {"image": str(image_path), "issues": []}
        img, err = _imread_unicode(image_path)
        if err:
            rec["issues"].append(f"损坏图片: {err}")
            stats["corrupted_image"] += 1
            stats["bad_samples"] += 1
            samples.append(rec)
            continue

        height, width = img.shape[:2]
        if width < 2 or height < 2:
            rec["issues"].append(f"图像尺寸异常: {width}x{height}")
            stats["bad_size"] += 1

        label_path = label_path_for(image_path, images_dir, labels_dir)
        boxes, parse_errs = parse_yolo_file(label_path)
        if parse_errs:
            rec["issues"].extend(parse_errs)
            if any("缺少标签" in e for e in parse_errs):
                stats["missing_label"] += 1
            else:
                stats["bad_label_file"] += 1

        box_issues = check_boxes(boxes, width, height, num_classes)
        rec["issues"].extend(box_issues)
        if box_issues:
            stats["abnormal_boxes"] += 1
        if not boxes and label_path.is_file() and not parse_errs:
            stats["empty_label"] += 1

        if rec["issues"]:
            samples.append(rec)
            stats["bad_samples"] += 1
        else:
            stats["ok_samples"] += 1

    stats["images_scanned"] = n_img
    return {"stats": dict(stats), "samples": samples}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 标注校验（越界 / 损坏图 / 异常样本）")
    p.add_argument("--root", type=Path, default=Path("data"), help="数据集根目录")
    p.add_argument("--images", type=Path, default=None)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--num-classes", type=int, default=len(DEFAULT_CLASSES))
    p.add_argument("--max-report", type=int, default=40, help="最多打印/写出的问题样本条数")
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--out-json", type=Path, default=None, help="可选：小型摘要 JSON（问题列表已截断）")
    p.add_argument("--no-recursive", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir, labels_dir = resolve_dirs(args.root, args.images, args.labels)
    if not images_dir.is_dir():
        raise SystemExit(
            f"图像目录不存在: {images_dir}\n"
            "公开仓库的 data/images 可能只有占位。请在本地放入图片后再运行。"
        )

    result = validate_dataset(
        images_dir,
        labels_dir,
        args.num_classes,
        recursive=not args.no_recursive,
        progress_every=args.progress_every,
    )
    stats = result["stats"]
    samples = result["samples"]
    shown = samples[: max(args.max_report, 0)]

    print("=== 校验摘要 ===")
    for k in sorted(stats):
        print(f"{k}: {stats[k]}")
    print(f"问题样本展示 {len(shown)}/{len(samples)}（--max-report 控制）")
    for rec in shown:
        print(f"- {rec['image']}")
        for issue in rec["issues"][:8]:
            print(f"    {issue}")

    if args.out_json:
        dump_json(
            args.out_json,
            {
                "stats": stats,
                "truncated": len(samples) > len(shown),
                "samples": shown,
            },
        )
        print(f"已写摘要: {args.out_json}")


if __name__ == "__main__":
    main()
