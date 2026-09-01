"""
用公开电力检测数据微调本仓库 YOLOv5sLite，导出 yolov5s_lite.pt。

初始化：官方 YOLOv5s COCO（yolov5s.pt）主干形状匹配层 + 随机 4 类检测头。
数据：scripts/prepare_public_power_data.py 产出的 4 类 YOLO 集。

说明：这是公开学术数据上的微调，覆盖绝缘子/缺陷（CPLID）与鸟巢/部分异物（FOTL），
不能等同于电网现场全量业务模型，但检测头已经过标注监督，不再是随机初始化。

用法::

    python scripts/prepare_public_power_data.py
    python scripts/train_yolov5s_lite.py --epochs 40 --imgsz 416 --batch 4
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viscale.detection.loss import ComputeLoss
from viscale.detection.transfer import load_coco_backbone
from viscale.detection.yolov5s_lite import POWER_SECURITY_CLASSES, YOLOv5sLite

COCO_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


def hsv_jitter(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-8, 8)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class YoloSet(Dataset):
    def __init__(self, data_root: Path, names: list[str], imgsz: int, augment: bool) -> None:
        self.imgsz = imgsz
        self.augment = augment
        img_dir = data_root / "images"
        lab_dir = data_root / "labels"
        self.items: list[tuple[Path, Path]] = []
        wanted = set(names) if names else None
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            if wanted is not None and img.name not in wanted:
                continue
            lab = lab_dir / f"{img.stem}.txt"
            if lab.is_file():
                self.items.append((img, lab))
        if not self.items:
            raise FileNotFoundError(f"no labeled images under {data_root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        img_path, lab_path = self.items[index]
        bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        h0, w0 = bgr.shape[:2]
        labels = []
        for line in lab_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid, xc, yc, bw, bh = (float(x) for x in parts[:5])
            labels.append([int(cid), xc, yc, bw, bh])
        if self.augment and random.random() < 0.5:
            bgr = cv2.flip(bgr, 1)
            for row in labels:
                row[1] = 1.0 - row[1]
        if self.augment:
            bgr = hsv_jitter(bgr)
        canvas, scale, (left, top) = letterbox(bgr, self.imgsz)
        tlabels = []
        for cid, xc, yc, bw, bh in labels:
            x = xc * w0 * scale + left
            y = yc * h0 * scale + top
            w = bw * w0 * scale
            h = bh * h0 * scale
            tlabels.append(
                [
                    cid,
                    x / self.imgsz,
                    y / self.imgsz,
                    w / self.imgsz,
                    h / self.imgsz,
                ]
            )
        rgb = canvas[:, :, ::-1].transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32) / 255.0)
        target = torch.tensor(tlabels, dtype=torch.float32) if tlabels else torch.zeros((0, 5))
        return tensor, target


def collate(batch):
    imgs, labs = zip(*batch)
    targets = []
    for i, lab in enumerate(labs):
        if lab.numel() == 0:
            continue
        img_id = torch.full((lab.shape[0], 1), i, dtype=torch.float32)
        targets.append(torch.cat((img_id, lab), 1))
    if targets:
        targets_t = torch.cat(targets, 0)
    else:
        targets_t = torch.zeros((0, 6), dtype=torch.float32)
    return torch.stack(imgs, 0), targets_t


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def download_coco(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print("[info] official yolov5s.pt already present")
        return
    print("[info] downloading official YOLOv5s COCO weights (ultralytics v7.0)")
    urllib.request.urlretrieve(COCO_URL, str(dest))
    print("[info] saved models/checkpoints/yolov5s.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv5sLite on public 4-class power data")
    p.add_argument("--data", type=str, default="data/public_yolo")
    p.add_argument("--coco", type=str, default="models/checkpoints/yolov5s.pt")
    p.add_argument("--out", type=str, default="models/checkpoints/yolov5s_lite.pt")
    p.add_argument("--attn", type=str, default="eca", choices=("eca", "se", "cbam", "none"))
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--max-images", type=int, default=0, help="0=use all; CPU 可先用 300 做短训")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = ROOT / args.data
    coco_path = ROOT / args.coco if not Path(args.coco).is_absolute() else Path(args.coco)
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    download_coco(coco_path)

    train_names = read_split(data_root / "splits" / "train.txt")
    val_names = read_split(data_root / "splits" / "val.txt")
    if args.max_images and train_names:
        train_names = train_names[: args.max_images]
        val_names = val_names[: max(1, args.max_images // 8)]
    train_set = YoloSet(data_root, train_names, args.imgsz, augment=True)
    val_set = YoloSet(data_root, val_names or train_names[: max(1, len(train_names) // 8)], args.imgsz, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} train={len(train_set)} val={len(val_set)}")

    model = YOLOv5sLite(num_classes=len(POWER_SECURITY_CLASSES), attn=args.attn)
    n_copied = load_coco_backbone(model, coco_path)
    print(f"[info] copied {n_copied} COCO backbone tensors; detect head starts random then fine-tunes")
    model.to(device)
    criterion = ComputeLoss(model.detect, model.num_classes)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 1))

    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss, _stats = criterion(preds, targets)
            if not torch.isfinite(loss):
                print("[warn] non-finite loss, skip batch")
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()
            running += float(loss.detach())
            n_batches += 1
        sched.step()
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                model.train()
                preds = model(imgs)
                loss, _ = criterion(preds, targets)
                model.eval()
                val_loss += float(loss)
                n_val += 1
        train_m = running / max(n_batches, 1)
        val_m = val_loss / max(n_val, 1)
        print(f"[epoch {epoch:03d}/{args.epochs}] train={train_m:.4f} val={val_m:.4f}")
        if val_m < best:
            best = val_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": best_state,
        "num_classes": len(POWER_SECURITY_CLASSES),
        "class_names": list(POWER_SECURITY_CLASSES),
        "attn": args.attn,
        "demo_adapter": False,
        "public_finetune": True,
        "data": "CPLID+FOTL_Drone mapped to 4 classes",
        "coco_backbone_tensors": n_copied,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "note": (
            "Fine-tuned on public CPLID (insulator/defect) and FOTL_Drone (nest/foreign). "
            "Not a private utility-company production model."
        ),
    }
    torch.save(payload, str(out_path))
    print(f"[info] wrote {out_path.as_posix()} (gitignored)")


if __name__ == "__main__":
    main()
