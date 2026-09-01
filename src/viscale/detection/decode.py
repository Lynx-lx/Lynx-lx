"""Box decode and class-agnostic NMS for the lite detector."""

from __future__ import annotations

import torch
from torch import Tensor


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def box_iou(a: Tensor, b: Tensor) -> Tensor:
    """IoU of boxes in xyxy, shapes (N,4) and (M,4) -> (N,M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def nms(boxes: Tensor, scores: Tensor, iou_thres: float) -> Tensor:
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thres]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def non_max_suppression(
    pred: Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> list[Tensor]:
    """
    pred: (B, N, 5+nc) with xywh in pixels, obj, class logits/probs.
    returns list of (M, 6) = xyxy, conf, cls
    """
    bs, _, dims = pred.shape
    nc = dims - 5
    out: list[Tensor] = []
    for b in range(bs):
        x = pred[b]
        obj = x[:, 4:5]
        cls = x[:, 5:]
        if nc:
            scores = obj * cls
            conf, cidx = scores.max(dim=1)
        else:
            conf, cidx = obj.squeeze(1), torch.zeros_like(obj.squeeze(1), dtype=torch.long)
        mask = conf > conf_thres
        x, conf, cidx = x[mask], conf[mask], cidx[mask]
        if x.numel() == 0:
            out.append(x.new_zeros((0, 6)))
            continue
        boxes = xywh_to_xyxy(x[:, :4])
        keep = nms(boxes, conf, iou_thres)[:max_det]
        det = torch.cat((boxes[keep], conf[keep, None], cidx[keep, None].float()), dim=1)
        out.append(det)
    return out
