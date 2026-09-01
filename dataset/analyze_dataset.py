"""数据集统计分析：样本数、类别框分布，适配万级规模。

只聚合计数，不写出逐图明细。可选写一份小型 JSON 摘要。

示例::

    python dataset/analyze_dataset.py --root data
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import (
    DEFAULT_CLASSES,
    dump_json,
    iter_image_files,
    label_path_for,
    load_class_names,
    parse_yolo_file,
    resolve_dirs,
)

# COCO 风格：小 / 中 / 大框（面积为归一化 w*h 相对整图）
SMALL_MAX = 32 * 32 / (640 * 640)
MEDIUM_MAX = 96 * 96 / (640 * 640)


def size_bucket(w: float, h: float) -> str:
    area = w * h
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def analyze(
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    *,
    recursive: bool = True,
    progress_every: int = 500,
) -> dict:
    n_images = 0
    n_with_label = 0
    n_missing_label = 0
    n_empty_label = 0
    n_parse_fail = 0
    box_by_class: Counter[int] = Counter()
    img_by_class: Counter[int] = Counter()  # 至少含该类一框的图像数
    size_hist: Counter[str] = Counter()
    boxes_per_image: list[int] = []

    for image_path in iter_image_files(images_dir, recursive=recursive):
        n_images += 1
        if progress_every and n_images % progress_every == 0:
            print(f"[analyze] 已统计 {n_images} 张 ...", flush=True)

        label_path = label_path_for(image_path, images_dir, labels_dir)
        boxes, errors = parse_yolo_file(label_path)
        if any("缺少标签" in e for e in errors):
            n_missing_label += 1
            boxes_per_image.append(0)
            continue
        if errors:
            n_parse_fail += 1
        if not boxes:
            n_empty_label += 1
            boxes_per_image.append(0)
            continue

        n_with_label += 1
        boxes_per_image.append(len(boxes))
        present: set[int] = set()
        for cls_id, _, _, w, h in boxes:
            box_by_class[cls_id] += 1
            present.add(cls_id)
            size_hist[size_bucket(w, h)] += 1
        for cid in present:
            img_by_class[cid] += 1

    n_boxes = sum(box_by_class.values())
    bpi = boxes_per_image
    mean_bpi = (sum(bpi) / len(bpi)) if bpi else 0.0

    per_class = []
    max_id = max(box_by_class.keys(), default=-1)
    n_names = max(len(class_names), max_id + 1)
    names = list(class_names) + [f"class_{i}" for i in range(len(class_names), n_names)]
    for cid in range(n_names):
        per_class.append(
            {
                "id": cid,
                "name": names[cid],
                "boxes": int(box_by_class[cid]),
                "images": int(img_by_class[cid]),
            }
        )

    return {
        "images": n_images,
        "images_with_boxes": n_with_label,
        "missing_label_files": n_missing_label,
        "empty_label_files": n_empty_label,
        "label_parse_failures": n_parse_fail,
        "boxes_total": n_boxes,
        "boxes_per_image_mean": round(mean_bpi, 4),
        "box_size": dict(size_hist),
        "per_class": per_class,
    }


def print_report(summary: dict) -> None:
    print("=== 数据集规模 ===")
    print(f"图像数: {summary['images']}")
    print(f"含有效框图像: {summary['images_with_boxes']}")
    print(f"缺标签文件: {summary['missing_label_files']}")
    print(f"空标签: {summary['empty_label_files']}")
    print(f"标签解析失败: {summary['label_parse_failures']}")
    print(f"框总数: {summary['boxes_total']}")
    print(f"平均每图框数: {summary['boxes_per_image_mean']}")
    print(f"框尺度 small/medium/large: {summary['box_size']}")
    print("=== 类别分布 ===")
    print(f"{'id':>4} {'name':<18} {'boxes':>10} {'images':>10}")
    for row in summary["per_class"]:
        print(f"{row['id']:>4} {row['name']:<18} {row['boxes']:>10} {row['images']:>10}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 数据集统计（类别分布 / 样本量）")
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--images", type=Path, default=None)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--names", type=Path, default=None, help="可选类别名文件，一行一个")
    p.add_argument("--out-json", type=Path, default=None, help="可选小型统计 JSON")
    p.add_argument("--progress-every", type=int, default=500)
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
    names = load_class_names(args.names)
    summary = analyze(
        images_dir,
        labels_dir,
        names,
        recursive=not args.no_recursive,
        progress_every=args.progress_every,
    )
    print_report(summary)
    if args.out_json:
        dump_json(args.out_json, summary)
        print(f"已写摘要: {args.out_json}")


if __name__ == "__main__":
    main()
