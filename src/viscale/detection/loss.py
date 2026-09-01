"""YOLOv5-style multi-head loss for YOLOv5sLite (training-time raw logits)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from viscale.detection.yolov5s_lite import Detect


def bbox_iou_xywh(pbox: Tensor, tbox: Tensor, eps: float = 1e-7) -> Tensor:
    """CIoU between (n,4) xywh boxes in the same coordinate frame."""
    px, py, pw, ph = pbox.unbind(-1)
    tx, ty, tw, th = tbox.unbind(-1)
    p1x, p1y = px - pw / 2, py - ph / 2
    p2x, p2y = px + pw / 2, py + ph / 2
    t1x, t1y = tx - tw / 2, ty - th / 2
    t2x, t2y = tx + tw / 2, ty + th / 2
    inter = (p2x.minimum(t2x) - p1x.maximum(t1x)).clamp(min=0) * (
        p2y.minimum(t2y) - p1y.maximum(t1y)
    ).clamp(min=0)
    union = pw * ph + tw * th - inter + eps
    iou = inter / union
    cw = p2x.maximum(t2x) - p1x.minimum(t1x)
    ch = p2y.maximum(t2y) - p1y.minimum(t1y)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = (px - tx).pow(2) + (py - ty).pow(2)
    v = (4 / (torch.pi**2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return iou - (rho2 / c2 + alpha * v)


class ComputeLoss:
    def __init__(self, detect: Detect, nc: int) -> None:
        self.detect = detect
        self.nc = nc
        self.na = detect.na
        self.nl = detect.nl
        self.anchor_t = 4.0
        self.box_gain = 0.05
        self.obj_gain = 1.0
        self.cls_gain = 0.5
        self.balance = [4.0, 1.0, 0.4, 0.1][: self.nl]
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, preds: list[Tensor], targets: Tensor) -> tuple[Tensor, dict[str, float]]:
        device = preds[0].device
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)
        for i, pred in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pred.shape[:4], device=device, dtype=pred.dtype)
            n = b.shape[0]
            if n:
                ps = pred[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou_xywh(pbox, tbox[i]).clamp(0, 1)
                lbox = lbox + (1.0 - iou).mean()
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
                if self.nc > 1:
                    t = torch.zeros((n, self.nc), device=device)
                    t[torch.arange(n, device=device), tcls[i]] = 1.0
                    lcls = lcls + self.bce(ps[:, 5:], t)
            obji = self.bce(pred[..., 4], tobj)
            lobj = lobj + obji * self.balance[i]
        n_layers = max(self.nl, 1)
        lbox = lbox * self.box_gain
        lobj = lobj * self.obj_gain
        lcls = lcls * self.cls_gain * (self.nc / 80.0 if self.nc else 1.0)
        total = (lbox + lobj + lcls) * n_layers
        stats = {
            "box": float(lbox.detach()),
            "obj": float(lobj.detach()),
            "cls": float(lcls.detach()),
        }
        return total, stats

    def build_targets(self, preds: list[Tensor], targets: Tensor):
        detect = self.detect
        na, nt = self.na, targets.shape[0]
        device = preds[0].device
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, max(nt, 1))
        if nt:
            targets_na = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        else:
            targets_na = torch.zeros((na, 0, 7), device=device)
        g = 0.5
        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=device,
            dtype=torch.float32,
        ) * g
        anchors_all = detect.anchors.to(device)
        for i in range(self.nl):
            anchors = anchors_all[i]
            ny, nx = preds[i].shape[2], preds[i].shape[3]
            gain[2:6] = torch.tensor([nx, ny, nx, ny], device=device, dtype=torch.float32)
            t = targets_na * gain
            if nt:
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1.0 / r.clamp(min=1e-8)).max(2)[0] < self.anchor_t
                t = t[j]
                if t.shape[0]:
                    gxy = t[:, 2:4]
                    gxi = gain[[2, 3]] - gxy
                    j1, k1 = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                    l1, m1 = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                    mask = torch.stack((torch.ones_like(j1), j1, k1, l1, m1))
                    t = t.repeat((5, 1, 1))[mask]
                    offsets = (torch.zeros_like(gxy)[None] + off[:, None])[mask]
                else:
                    offsets = 0
                    t = t
            else:
                t = targets_na[0]
                offsets = 0
            if t.numel() == 0:
                indices.append(
                    (
                        torch.zeros(0, dtype=torch.long, device=device),
                        torch.zeros(0, dtype=torch.long, device=device),
                        torch.zeros(0, dtype=torch.long, device=device),
                        torch.zeros(0, dtype=torch.long, device=device),
                    )
                )
                tbox.append(torch.zeros(0, 4, device=device))
                anch.append(torch.zeros(0, 2, device=device))
                tcls.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            bc = t[:, 0:2]
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            a = t[:, 6].long()
            b, c = bc.long().T
            gij = (gxy - offsets).long()
            gi = gij[:, 0].clamp(0, nx - 1)
            gj = gij[:, 1].clamp(0, ny - 1)
            indices.append((b, a, gj, gi))
            tbox.append(torch.cat((gxy - gij.float(), gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        return tcls, tbox, indices, anch
