"""本地数据集布局约定与轻量扫描（不包含真实图像）。

data/ 在公开仓库中只有 .gitkeep。请在本地自行放入数据，不要 git add 原始图片。

预期目录（YOLO 检测）::

    data/
      images/          # 训练/推理图像：.jpg .png .bmp ...
      labels/          # 与图像同名的 .txt
                       # 每行: class_id  xc  yc  w  h   （均相对宽高归一化到 0~1）
      raw/             # 可选：未处理原图/视频
      processed/       # 可选：预处理缓存
      annotations/     # 可选：其它标注导出
      samples/         # 可选：少量可公开样例（仍建议不提交大文件）

类别 id 与 ``viscale.detection.POWER_SECURITY_CLASSES`` 顺序一致（从 0 起）。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / "data"


def describe_layout() -> str:
    return __doc__ or ""


def data_root_status(root: Path | None = None) -> str:
    """检查 data 是否存在；缺少子目录时给出放置说明，不抛异常。"""
    base = root or DEFAULT_DATA_ROOT
    images = base / "images"
    labels = base / "labels"
    lines = [f"数据根目录: {base}"]
    if not base.is_dir():
        lines.append("目录不存在。请在仓库根创建 data/images 与 data/labels 后放入本地数据。")
        return "\n".join(lines)
    if not images.is_dir():
        lines.append("缺少 data/images/ 。请将图像放到该文件夹（仓库不收录原始数据集）。")
    else:
        n = sum(1 for p in images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        lines.append(f"images: 约 {n} 张（仅统计常见后缀）")
    if not labels.is_dir():
        lines.append("缺少 data/labels/ 。YOLO txt 应与图像同名。")
    return "\n".join(lines)
