"""按比例划分 train / val / test，只写路径清单，不复制图像。

默认 7:2:1。可按图像中最稀有类别做分层，减轻万级长尾类别被分空的问题。

示例::

    python dataset/split_dataset.py --root data --ratios 0.7,0.2,0.1 --out-dir dataset/splits
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    iter_image_files,
    label_path_for,
    parse_yolo_file,
    resolve_dirs,
)


def parse_ratios(text: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("ratios 必须是 train,val,test 三个数，例如 0.7,0.2,0.1")
    s = sum(parts)
    if s <= 0:
        raise ValueError("ratios 之和须为正")
    return tuple(x / s for x in parts)  # type: ignore[return-value]


def image_stratum(image_path: Path, images_dir: Path, labels_dir: Path) -> str:
    """分层键：该图出现的最小 class_id（稀有类优先成组）；无框为 empty。"""
    boxes, errs = parse_yolo_file(label_path_for(image_path, images_dir, labels_dir))
    if errs and not boxes:
        return "invalid"
    if not boxes:
        return "empty"
    return str(min(b[0] for b in boxes))


def split_groups(
    groups: dict[str, list[Path]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Path]]:
    """各组按同一比例切分，再合并。余数从前往后补齐，保证每张图只出现一次。"""
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    r_train, r_val, r_test = ratios

    for _, items in sorted(groups.items(), key=lambda kv: kv[0]):
        order = np.array(items, dtype=object)
        rng.shuffle(order)
        n = len(order)
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        n_test = int(n * r_test)
        # 舍入缺口补到训练集，避免样本丢失
        remain = n - (n_train + n_val + n_test)
        n_train += remain
        buckets["train"].extend(order[:n_train].tolist())
        buckets["val"].extend(order[n_train : n_train + n_val].tolist())
        buckets["test"].extend(order[n_train + n_val : n_train + n_val + n_test].tolist())
    return buckets


def write_list(path: Path, images: list[Path], images_dir: Path) -> None:
    """每行一条相对 images_dir 的 POSIX 路径，便于跨平台训练脚本读取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    root = images_dir.resolve()
    for p in images:
        try:
            rel = p.resolve().relative_to(root).as_posix()
        except ValueError:
            rel = p.resolve().as_posix()
        lines.append(rel)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="划分 YOLO 数据集（仅输出路径清单）")
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--images", type=Path, default=None)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--ratios", type=str, default="0.7,0.2,0.1", help="train,val,test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("dataset/splits"))
    p.add_argument("--no-stratify", action="store_true", help="关闭按类别分层，改为全局随机")
    p.add_argument("--no-recursive", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios)
    images_dir, labels_dir = resolve_dirs(args.root, args.images, args.labels)
    if not images_dir.is_dir():
        raise SystemExit(f"图像目录不存在: {images_dir}")

    paths = list(iter_image_files(images_dir, recursive=not args.no_recursive))
    if not paths:
        raise SystemExit(f"未找到图像: {images_dir}")

    groups: dict[str, list[Path]] = defaultdict(list)
    if args.no_stratify:
        groups["all"] = paths
    else:
        for i, img in enumerate(paths, start=1):
            if i % 1000 == 0:
                print(f"[split] 读取标签分层 {i}/{len(paths)}", flush=True)
            groups[image_stratum(img, images_dir, labels_dir)].append(img)

    split = split_groups(dict(groups), ratios, args.seed)
    for name, items in split.items():
        write_list(args.out_dir / f"{name}.txt", items, images_dir)

    print("=== 划分结果（仅清单，未复制文件）===")
    total = sum(len(v) for v in split.values())
    for name in ("train", "val", "test"):
        n = len(split[name])
        pct = 100.0 * n / total if total else 0.0
        print(f"{name}: {n} ({pct:.1f}%) -> {args.out_dir / (name + '.txt')}")
    print(f"分层组数: {len(groups)}  seed={args.seed}")


if __name__ == "__main__":
    main()
