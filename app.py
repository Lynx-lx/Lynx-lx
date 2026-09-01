"""本地 Gradio 演示：上传图片 → YOLOv5sLite 检测 → 风险评估。

启动（项目根目录）::

    pip install -r requirements.txt
    python app.py

权重路径通过 --weights 或环境变量 VISCALE_WEIGHTS 预留；文件不存在则随机初始化。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from viscale.detection import POWER_SECURITY_CLASSES, build_yolov5s_lite
from viscale.io.camera import DEFAULT_LOCAL_CAMERA, load_camera_config
from viscale.measurement.scale import meters_per_pixel_at_depth
from viscale.risk import DetectionRecord, RiskAssessor

# ---------- 预留配置（可被 CLI / 环境变量覆盖）----------
DEFAULT_WEIGHTS = os.environ.get("VISCALE_WEIGHTS", str(ROOT / "models" / "checkpoints" / "yolov5s_lite.pt"))
DEFAULT_ATTN = os.environ.get("VISCALE_ATTN", "eca")
DEFAULT_DEVICE = os.environ.get("VISCALE_DEVICE", "cpu")
DEFAULT_IMGSZ = int(os.environ.get("VISCALE_IMGSZ", "640"))

BOX_COLORS = [
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (230, 126, 34),
    (155, 89, 182),
    (231, 76, 60),
    (26, 188, 156),
    (149, 165, 166),
]
GRADE_COLOR = {1: "#27ae60", 2: "#2980b9", 3: "#e67e22", 4: "#c0392b"}

_lock = threading.Lock()
_runtime: dict = {
    "model": None,
    "device": None,
    "assessor": None,
    "weights_note": "",
    "camera_note": "",
    "mpp_from_camera": None,
}


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _load_weights(model: torch.nn.Module, path: str, device: torch.device) -> str:
    if not path:
        return "未配置权重，当前为随机初始化（演示框可能无意义）"
    p = Path(path)
    if not p.is_file():
        return f"权重文件不存在，已跳过加载: {p}（随机初始化）"
    ckpt = torch.load(str(p), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    return f"已加载权重: {p}"


def init_runtime(
    weights: str,
    attn: str,
    device_name: str,
    camera_config: str | None = None,
    working_distance_m: float = 0.0,
) -> None:
    with _lock:
        device = _resolve_device(device_name)
        model = build_yolov5s_lite(num_classes=len(POWER_SECURITY_CLASSES), attn=attn).to(device)
        model.eval()
        note = _load_weights(model, weights, device)
        cam_path = camera_config or str(DEFAULT_LOCAL_CAMERA)
        cam, cam_msg = load_camera_config(cam_path)
        print(cam_msg)
        mpp = meters_per_pixel_at_depth(cam, working_distance_m) if working_distance_m > 0 else None
        _runtime["model"] = model
        _runtime["device"] = device
        _runtime["assessor"] = RiskAssessor()
        _runtime["weights_note"] = note
        _runtime["camera_note"] = cam_msg.split("\n")[0]
        _runtime["mpp_from_camera"] = mpp
        print(note)
        print(f"device={device} params={model.parameter_count() / 1e6:.2f}M attn={attn}")


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


def to_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = image_bgr[:, :, ::-1]
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).to(device)


def map_xyxy(xyxy: list[float], pad: tuple[int, int], scale: float, wh: tuple[int, int]) -> tuple[int, int, int, int]:
    left, top = pad
    w, h = wh
    x1 = int(np.clip((xyxy[0] - left) / scale, 0, w - 1))
    y1 = int(np.clip((xyxy[1] - top) / scale, 0, h - 1))
    x2 = int(np.clip((xyxy[2] - left) / scale, 0, w - 1))
    y2 = int(np.clip((xyxy[3] - top) / scale, 0, h - 1))
    return x1, y1, x2, y2


def draw_boxes(image_rgb: np.ndarray, rows: list[dict]) -> np.ndarray:
    vis = image_rgb.copy()
    bgr = vis[:, :, ::-1].copy()
    for row in rows:
        x1, y1, x2, y2 = row["xyxy"]
        cid = int(row["cls_id"])
        color = BOX_COLORS[cid % len(BOX_COLORS)]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{row['cls_name']} {row['conf']:.2f}"
        cv2.putText(bgr, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bgr[:, :, ::-1]


def as_rgb(image) -> np.ndarray | None:
    """接受 Gradio numpy / 路径 / PIL。返回 RGB uint8。"""
    if image is None:
        return None
    if isinstance(image, (str, Path)):
        data = np.fromfile(str(image), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def as_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def run_infer(image, conf: float, iou: float, meters_per_pixel: float, imgsz: int):
    rgb = as_rgb(image)
    if rgb is None:
        return None, "请先上传本地图片。", []

    model = _runtime["model"]
    device = _runtime["device"]
    assessor: RiskAssessor = _runtime["assessor"]
    names = list(model.class_names)
    image_bgr = as_bgr(rgb)
    h, w = image_bgr.shape[:2]
    canvas, scale, pad = letterbox(image_bgr, int(imgsz))
    tensor = to_tensor(canvas, device)

    with torch.inference_mode():
        dets = model.predict(tensor, conf_thres=float(conf), iou_thres=float(iou))[0]

    rows: list[dict] = []
    records: list[DetectionRecord] = []
    for *xyxy, score, cls_id in dets.cpu().tolist():
        mapped = map_xyxy(xyxy, pad, scale, (w, h))
        cid = int(cls_id)
        name = names[cid] if 0 <= cid < len(names) else str(cid)
        rows.append({"cls_id": cid, "cls_name": name, "conf": float(score), "xyxy": mapped})
        records.append(DetectionRecord(cls_name=name, conf=float(score), xyxy=tuple(float(v) for v in mapped)))

    mpp = float(meters_per_pixel) if meters_per_pixel and meters_per_pixel > 0 else _runtime.get("mpp_from_camera")
    report = assessor.assess(records, meters_per_pixel=mpp, image_wh=(w, h), update_elc=False)

    vis = draw_boxes(rgb, rows)
    color = GRADE_COLOR.get(int(report.grade), "#333")
    viol_lines = "无"
    if report.violations:
        viol_lines = "\n".join(
            f"- **{ev.kind}** 分数 {ev.score:.2f}：{ev.message}" for ev in report.violations
        )
    cam_line = _runtime.get("camera_note") or ""
    md = (
        f"<div style='padding:12px;border-radius:8px;border:1px solid {color};'>"
        f"<p style='margin:0;color:{color};font-size:22px;font-weight:700;'>风险等级：{report.label_zh}</p>"
        f"<p style='margin:8px 0 0;'>综合分数 <b>{report.score:.3f}</b>"
        f" ｜ 等级 {int(report.grade)} / 4"
        f" ｜ ELC 阈值 {[round(t, 3) for t in report.thresholds]}</p>"
        f"<p style='margin:8px 0 0;color:#666;font-size:13px;'>{_runtime['weights_note']}</p>"
        f"<p style='margin:8px 0 0;color:#666;font-size:13px;'>{cam_line}</p>"
        f"</div>\n\n**违规项**\n\n{viol_lines}\n\n"
        f"**检测框数量：** {len(rows)}"
    )
    table = [
        [r["cls_name"], f"{r['conf']:.3f}", r["xyxy"][0], r["xyxy"][1], r["xyxy"][2], r["xyxy"][3]]
        for r in rows
    ]
    return vis, md, table


def build_ui(imgsz: int):
    import gradio as gr

    with gr.Blocks(title="电力安防 · 检测与风险评估") as demo:
        gr.Markdown(
            "## 电力安防视觉检测演示\n"
            "上传本地图片，运行轻量化 YOLOv5s（P2 小目标分支 + 注意力）并输出风险等级。\n"
            "适合本机演示；无权重时为随机初始化。真实数据集与相机标定请放在本地，仓库仅含代码与模板。"
        )
        with gr.Row():
            inp = gr.Image(type="filepath", label="上传图片", sources=["upload"])
            out = gr.Image(type="numpy", label="检测框可视化")
        with gr.Row():
            conf = gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="置信度阈值")
            iou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU")
            mpp = gr.Number(value=0.0, label="米/像素（0 表示不启用尺度）")
        btn = gr.Button("检测并评估风险", variant="primary")
        risk = gr.Markdown()
        table = gr.Dataframe(
            headers=["类别", "置信度", "x1", "y1", "x2", "y2"],
            label="检测列表",
            interactive=False,
        )
        btn.click(
            fn=lambda im, c, u, m: run_infer(im, c, u, m, imgsz),
            inputs=[inp, conf, iou, mpp],
            outputs=[out, risk, table],
            api_name="infer",
        )
    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio 本地检测 + 风险评估演示")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="预留权重路径 .pt")
    p.add_argument("--attn", type=str, default=DEFAULT_ATTN, choices=("eca", "se", "cbam", "none"))
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument(
        "--camera-config",
        type=str,
        default=str(DEFAULT_LOCAL_CAMERA),
        help="本地相机 YAML（默认 config/camera_sensor/camera.yaml；不存在则提示并跳过）",
    )
    p.add_argument(
        "--working-distance-m",
        type=float,
        default=0.0,
        help="目标大致深度（米），与真实内参一起估算米/像素；0 表示不估算",
    )
    return p.parse_args()


def main() -> None:
    import gradio as gr

    args = parse_args()
    init_runtime(
        weights=args.weights,
        attn=args.attn,
        device_name=args.device,
        camera_config=args.camera_config,
        working_distance_m=args.working_distance_m,
    )
    demo = build_ui(args.imgsz)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
