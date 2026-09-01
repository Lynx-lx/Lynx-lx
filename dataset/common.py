"""数据集脚本共用工具：YOLO 目录约定、标签解析、万级遍历。

公开仓库的 data/ 仅有空目录占位（.gitkeep），不含原始图片。

本地预期布局::

    data/
      images/    # .jpg/.png/...  与 labels 按相对路径同名
      labels/    # 每行: class_id xc yc w h （归一化 0~1，YOLO）
      raw/ processed/ annotations/ samples/   # 可选

也可把任意根目录传给 --root，只要其下有 images/ 与 labels/。
不在本模块内写入或复制图像数据。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

# 与 src/viscale/detection 中的类别顺序对齐（class_id 从 0 起）
DEFAULT_CLASSES: tuple[str, ...] = (
    "person",
    "helmet",
    "insulator",
    "bird_nest",
    "smoke",
    "fire",
    "vehicle",
    "foreign_object",
)

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def load_class_names(names_file: str | Path | None, num_hint: int | None = None) -> list[str]:
    """从文本文件读取类别名（一行一个）；未提供时用默认电力安防类别。"""
    if names_file:
        path = Path(names_file)
        names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not names:
            raise ValueError(f"类别文件为空: {path}")
        return names
    names = list(DEFAULT_CLASSES)
    if num_hint and num_hint > len(names):
        names.extend(f"class_{i}" for i in range(len(names), num_hint))
    return names


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def iter_image_files(images_dir: Path, *, recursive: bool = True) -> Iterator[Path]:
    """流式列出图像路径，避免一次性把万级路径全部读进复杂结构以外的冗余拷贝。"""
    pattern = "**/*" if recursive else "*"
    for path in images_dir.glob(pattern):
        if is_image_file(path):
            yield path


def label_path_for(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    """图像相对 images_dir 的路径，改到 labels_dir 下并换扩展名为 .txt。"""
    rel = image_path.resolve().relative_to(images_dir.resolve())
    return (labels_dir / rel).with_suffix(".txt")


def parse_yolo_line(line: str, line_no: int) -> tuple[int, float, float, float, float] | str:
    """解析一行 YOLO 标签。成功返回 (cls, xc, yc, w, h)，失败返回错误说明。"""
    parts = line.strip().split()
    if not parts:
        return f"第 {line_no} 行为空"
    if len(parts) < 5:
        return f"第 {line_no} 列数不足 5: {line.strip()!r}"
    try:
        cls_id = int(float(parts[0]))
        xc, yc, w, h = (float(parts[i]) for i in range(1, 5))
    except ValueError:
        return f"第 {line_no} 数值解析失败: {line.strip()!r}"
    return cls_id, xc, yc, w, h


def parse_yolo_file(label_path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """读取整个标签文件。文件不存在则 boxes 为空并带一条错误。"""
    if not label_path.is_file():
        return [], [f"缺少标签文件: {label_path}"]
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [f"无法读取标签: {label_path} ({exc})"]
    except UnicodeDecodeError:
        return [], [f"标签编码损坏: {label_path}"]

    boxes: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parsed = parse_yolo_line(raw, i)
        if isinstance(parsed, str):
            errors.append(parsed)
        else:
            boxes.append(parsed)
    return boxes, errors


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, width: int, height: int) -> tuple[float, float, float, float]:
    """归一化中心框转像素 xyxy，用于越界判断。"""
    bw, bh = w * width, h * height
    cx, cy = xc * width, yc * height
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def resolve_dirs(root: Path, images: Path | None, labels: Path | None) -> tuple[Path, Path]:
    """解析图像/标签目录：可显式传入，或使用 <root>/images 与 <root>/labels。

    公开仓库中这些目录通常只有 .gitkeep；本地放入真实图片后再跑脚本。
    """
    images_dir = images if images is not None else root / "images"
    labels_dir = labels if labels is not None else root / "labels"
    return images_dir, labels_dir


def dump_json(path: Path, payload: dict) -> None:
    """写入小型 JSON 摘要（调用方须保证内容不是逐框全量转储）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
