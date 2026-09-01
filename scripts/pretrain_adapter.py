"""
YOLOv5s COCO 主干 → 本仓库 YOLOv5s‑P2+注意力（4 类检测头）适配导出脚本。

【重要说明】此 demo 权重仅仅网络结构完整；检测头随机初始化，未在电力标注
数据集做微调，不能真实识别鸟巢、绝缘子缺陷；仅用于跑通整个推理链路，
不可以用于实际安防业务。

本仓库不使用电力标注集训练，因此无法得到能识别 insulator / bird_nest /
foreign_object / damaged_insulator 的业务权重。本脚本只保证工程代码完整、
可供 GitHub 面试仓库展示。

不把任何 .pt 打进 git。请在本地运行本脚本后，于 models/checkpoints/ 得到
yolov5s_lite_demo.pt（该文件已被 .gitignore 忽略）。

用法（项目根目录）::

    # 1. 自行下载官方 YOLOv5s COCO 权重，放到:
    #    models/checkpoints/yolov5s.pt
    #    （例如 ultralytics/yolov5 发布的 yolov5s.pt）
    # 2. 运行:
    python scripts/pretrain_adapter.py

流程：
  ① 构建本项目 4 类 YOLOv5sLite（P2 小目标头 + ECA 等注意力）
  ② 若存在 yolov5s.pt，按 tensor 形状尽量拷贝主干卷积；拷不进的层保持随机
  ③ 检测头 detect.* 一律不拷贝，保持随机初始化（未电力微调）
  ④ 导出 models/checkpoints/yolov5s_lite_demo.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from viscale.detection.transfer import torch_load, transfer_backbone, unwrap_state_dict
from viscale.detection.yolov5s_lite import POWER_SECURITY_CLASSES, YOLOv5sLite

COCO_SRC_REL = "models/checkpoints/yolov5s.pt"
DEMO_OUT_REL = "models/checkpoints/yolov5s_lite_demo.pt"
NUM_CLASSES = len(POWER_SECURITY_CLASSES)  # 4: insulator, bird_nest, foreign_object, damaged_insulator

LIMITATION = (
    "【重要说明】此demo权重仅仅网络结构完整；检测头随机初始化，未在电力标注数据集做微调，"
    "不能真实识别鸟巢、绝缘子缺陷；仅用于跑通整个推理链路，不可以用于实际安防业务。"
)


def build_adapted_model(attn: str = "eca") -> YOLOv5sLite:
    """4 类电力标签空间；P2 小目标分支 + 注意力已在 YOLOv5sLite 内。"""
    return YOLOv5sLite(num_classes=NUM_CLASSES, attn=attn)


def export_demo_checkpoint(coco_src: Path, out_path: Path, attn: str) -> None:
    print(LIMITATION)
    model = build_adapted_model(attn=attn)
    dst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_backbone = 0
    coco_ok = False

    if coco_src.is_file() and coco_src.stat().st_size > 0:
        print("[info] found COCO yolov5s.pt under models/checkpoints/; transferring backbone tensors by shape")
        raw = torch_load(coco_src, "cpu")
        src = unwrap_state_dict(raw)
        dst, n_backbone = transfer_backbone(src, dst)
        coco_ok = True
        print(f"[info] copied {n_backbone} backbone tensors; detect head kept randomly initialized")
    else:
        print("[warn] models/checkpoints/yolov5s.pt not found; export random full net (head still random)")
        print("[warn] download official yolov5s.pt yourself into models/checkpoints/ then re-run this script")

    model.load_state_dict(dst, strict=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "num_classes": NUM_CLASSES,
        "class_names": list(POWER_SECURITY_CLASSES),
        "attn": attn,
        "demo_adapter": True,
        "coco_backbone_partial": coco_ok,
        "backbone_tensors_copied": n_backbone,
        "note": LIMITATION,
    }
    torch.save(payload, str(out_path))
    print("[info] wrote models/checkpoints/yolov5s_lite_demo.pt (gitignored)")
    print(LIMITATION)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adapt public YOLOv5s backbone onto 4-class P2 lite head (demo only)")
    p.add_argument("--coco", type=str, default=COCO_SRC_REL, help="relative path to official yolov5s.pt")
    p.add_argument("--out", type=str, default=DEMO_OUT_REL)
    p.add_argument("--attn", type=str, default="eca", choices=("eca", "se", "cbam", "none"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    coco = Path(args.coco)
    out = Path(args.out)
    if not coco.is_absolute():
        coco = ROOT / coco
    if not out.is_absolute():
        out = ROOT / out
    export_demo_checkpoint(coco, out, args.attn)


if __name__ == "__main__":
    main()
