"""Lightweight YOLOv5s with a P2 small-object head and attention C3 blocks."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn.functional import interpolate

from viscale.detection.blocks import C3, SPPF, Conv
from viscale.detection.decode import non_max_suppression

POWER_SECURITY_CLASSES = (
    "insulator",
    "bird_nest",
    "foreign_object",
    "damaged_insulator",
)

# P2/4 extra-small, P3/8 small, P4/16 medium, P5/32 large
DEFAULT_ANCHORS = (
    (5, 6, 8, 14, 15, 11),
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)


def make_divisible(value: float, divisor: int = 8) -> int:
    return int(round(value / divisor) * divisor) or divisor


def scale_depth(n: int, depth: float) -> int:
    return max(round(n * depth), 1) if n > 1 else n


class Detect(nn.Module):
    def __init__(
        self,
        nc: int,
        anchors: Sequence[Sequence[int]],
        ch: Sequence[int],
        strides: Sequence[int],
    ) -> None:
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.stride = torch.tensor(strides, dtype=torch.float32)
        anchor_t = torch.tensor(anchors, dtype=torch.float32).view(self.nl, self.na, 2)
        self.register_buffer("anchors", anchor_t)
        self.heads = nn.ModuleList(nn.Conv2d(c, self.no * self.na, 1) for c in ch)
        self._init_bias()

    def _init_bias(self) -> None:
        for head, stride in zip(self.heads, self.stride.tolist()):
            b = head.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / stride) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc + 1e-6))
            head.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor] | torch.Tensor:
        outputs = []
        decoded = []
        for i, feat in enumerate(feats):
            pred = self.heads[i](feat)
            bs, _, ny, nx = pred.shape
            pred = pred.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(pred)
            if not self.training:
                decoded.append(self._decode(pred, i, ny, nx))
        if self.training:
            return outputs
        return torch.cat(decoded, dim=1)

    def _decode(self, pred: torch.Tensor, layer: int, ny: int, nx: int) -> torch.Tensor:
        device = pred.device
        stride = self.stride.to(device)[layer]
        gy, gx = torch.meshgrid(
            torch.arange(ny, device=device),
            torch.arange(nx, device=device),
            indexing="ij",
        )
        grid = torch.stack((gx, gy), dim=2).view(1, 1, ny, nx, 2).float()
        anchors = (self.anchors.to(device)[layer] * stride).view(1, self.na, 1, 1, 2)
        y = pred.sigmoid()
        xy = (y[..., 0:2] * 2 - 0.5 + grid) * stride
        wh = (y[..., 2:4] * 2) ** 2 * anchors
        return torch.cat((xy, wh, y[..., 4:]), dim=-1).view(pred.shape[0], -1, self.no)


class YOLOv5sLite(nn.Module):
    """
    YOLOv5s width/depth, plus:
      - P2/stride-4 detection branch for small objects
      - ECA/SE/CBAM on C3 outputs (default ECA)
    """

    def __init__(
        self,
        num_classes: int = len(POWER_SECURITY_CLASSES),
        depth: float = 0.33,
        width: float = 0.50,
        attn: str = "eca",
        anchors: Sequence[Sequence[int]] = DEFAULT_ANCHORS,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.attn = attn
        self.class_names = list(POWER_SECURITY_CLASSES[:num_classes])
        while len(self.class_names) < num_classes:
            self.class_names.append(f"class_{len(self.class_names)}")

        def c(base: int) -> int:
            return make_divisible(base * width, 8)

        n = lambda k: scale_depth(k, depth)

        # backbone
        self.stem = Conv(3, c(64), 6, 2, 2)  # P1/2
        self.b_p2_down = Conv(c(64), c(128), 3, 2)  # P2/4
        self.b_p2 = C3(c(128), c(128), n(3), attn=attn)
        self.b_p3_down = Conv(c(128), c(256), 3, 2)  # P3/8
        self.b_p3 = C3(c(256), c(256), n(6), attn=attn)
        self.b_p4_down = Conv(c(256), c(512), 3, 2)  # P4/16
        self.b_p4 = C3(c(512), c(512), n(9), attn=attn)
        self.b_p5_down = Conv(c(512), c(1024), 3, 2)  # P5/32
        self.b_p5 = C3(c(1024), c(1024), n(3), attn=attn)
        self.sppf = SPPF(c(1024), c(1024), 5)

        # FPN top-down (P5 -> P2)
        self.n_p5_lat = Conv(c(1024), c(512), 1, 1)
        self.n_p4 = C3(c(512) + c(512), c(512), n(3), shortcut=False, attn=attn)
        self.n_p4_lat = Conv(c(512), c(256), 1, 1)
        self.n_p3 = C3(c(256) + c(256), c(256), n(3), shortcut=False, attn=attn)
        self.n_p3_lat = Conv(c(256), c(128), 1, 1)
        self.n_p2 = C3(c(128) + c(128), c(128), n(3), shortcut=False, attn=attn)

        # PAN bottom-up (P2 -> P5)
        self.n_p2_down = Conv(c(128), c(128), 3, 2)
        self.n_p3_out = C3(c(128) + c(256), c(256), n(3), shortcut=False, attn=attn)
        self.n_p3_down = Conv(c(256), c(256), 3, 2)
        self.n_p4_out = C3(c(256) + c(512), c(512), n(3), shortcut=False, attn=attn)
        self.n_p4_down = Conv(c(512), c(512), 3, 2)
        self.n_p5_out = C3(c(512) + c(512), c(1024), n(3), shortcut=False, attn=attn)

        ch = (c(128), c(256), c(512), c(1024))
        self.detect = Detect(num_classes, anchors, ch, strides=(4, 8, 16, 32))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor] | torch.Tensor:
        x = self.stem(x)
        p2 = self.b_p2(self.b_p2_down(x))
        p3 = self.b_p3(self.b_p3_down(p2))
        p4 = self.b_p4(self.b_p4_down(p3))
        p5 = self.sppf(self.b_p5(self.b_p5_down(p4)))

        p5_lat = self.n_p5_lat(p5)
        p4_td = self.n_p4(torch.cat((interpolate(p5_lat, scale_factor=2, mode="nearest"), p4), 1))
        p4_lat = self.n_p4_lat(p4_td)
        p3_td = self.n_p3(torch.cat((interpolate(p4_lat, scale_factor=2, mode="nearest"), p3), 1))
        p3_lat = self.n_p3_lat(p3_td)
        p2_out = self.n_p2(torch.cat((interpolate(p3_lat, scale_factor=2, mode="nearest"), p2), 1))

        p3_out = self.n_p3_out(torch.cat((self.n_p2_down(p2_out), p3_td), 1))
        p4_out = self.n_p4_out(torch.cat((self.n_p3_down(p3_out), p4_td), 1))
        p5_out = self.n_p5_out(torch.cat((self.n_p4_down(p4_out), p5_lat), 1))
        return self.detect([p2_out, p3_out, p4_out, p5_out])

    @torch.inference_mode()
    def predict(
        self,
        x: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> list[torch.Tensor]:
        self.eval()
        pred = self.forward(x)
        assert isinstance(pred, torch.Tensor)
        return non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_yolov5s_lite(
    num_classes: int = len(POWER_SECURITY_CLASSES),
    attn: str = "eca",
) -> YOLOv5sLite:
    return YOLOv5sLite(num_classes=num_classes, attn=attn)
