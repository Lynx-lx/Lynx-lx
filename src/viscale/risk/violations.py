"""Custom violation evaluators for power-security scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

ViolationFn = Callable[["SceneContext"], list["ViolationEvent"]]
_REGISTRY: dict[str, ViolationFn] = {}


@dataclass
class DetectionRecord:
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


@dataclass
class ViolationEvent:
    kind: str
    score: float
    message: str
    refs: list[int] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class SceneContext:
    detections: list[DetectionRecord]
    meters_per_pixel: float | None = None
    image_wh: tuple[int, int] | None = None
    extra: dict = field(default_factory=dict)


def register_violation(name: str) -> Callable[[ViolationFn], ViolationFn]:
    def deco(fn: ViolationFn) -> ViolationFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def registered_violations() -> dict[str, ViolationFn]:
    return dict(_REGISTRY)


def _boxes(dets: Sequence[DetectionRecord]) -> np.ndarray:
    if not dets:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array([d.xyxy for d in dets], dtype=np.float64)


def _centers(xyxy: np.ndarray) -> np.ndarray:
    return np.stack(((xyxy[:, 0] + xyxy[:, 2]) * 0.5, (xyxy[:, 1] + xyxy[:, 3]) * 0.5), axis=1)


def _heights(xyxy: np.ndarray) -> np.ndarray:
    return np.maximum(xyxy[:, 3] - xyxy[:, 1], 1.0)


def _areas(xyxy: np.ndarray) -> np.ndarray:
    return np.maximum(xyxy[:, 2] - xyxy[:, 0], 0) * np.maximum(xyxy[:, 3] - xyxy[:, 1], 0)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _indices(dets: Sequence[DetectionRecord], name: str) -> list[int]:
    return [i for i, d in enumerate(dets) if d.cls_name == name]


def _pair_min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.full((len(a),), np.inf)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return d.min(axis=1)


@register_violation("ppe_missing_helmet")
def eval_ppe_missing_helmet(ctx: SceneContext) -> list[ViolationEvent]:
    """Person without a helmet whose center falls near the head region."""
    dets = ctx.detections
    p_idx = _indices(dets, "person")
    h_idx = _indices(dets, "helmet")
    if not p_idx:
        return []
    people = _boxes([dets[i] for i in p_idx])
    p_cent = _centers(people)
    p_h = _heights(people)
    helmets = _boxes([dets[i] for i in h_idx])
    h_cent = _centers(helmets) if len(h_idx) else np.zeros((0, 2))
    events: list[ViolationEvent] = []
    for j, pi in enumerate(p_idx):
        person = dets[pi]
        head_y = people[j, 1] + 0.25 * p_h[j]
        if len(h_idx):
            dist = np.linalg.norm(h_cent - np.array([p_cent[j, 0], head_y]), axis=1)
            nearest = float(dist.min())
            matched = nearest < 0.55 * p_h[j]
        else:
            nearest, matched = float("inf"), False
        if matched:
            continue
        events.append(
            ViolationEvent(
                kind="ppe_missing_helmet",
                score=_clip01(0.35 + 0.65 * person.conf),
                message="作业人员未检测到有效安全帽匹配",
                refs=[pi],
                extra={"head_gap_px": nearest if np.isfinite(nearest) else None},
            )
        )
    return events


@register_violation("nest_near_insulator")
def eval_nest_near_insulator(ctx: SceneContext) -> list[ViolationEvent]:
    """Bird nest risk rises when close to insulators (pixel or metric scale)."""
    dets = ctx.detections
    n_idx = _indices(dets, "bird_nest")
    i_idx = _indices(dets, "insulator")
    if not n_idx:
        return []
    nests = _boxes([dets[i] for i in n_idx])
    n_cent = _centers(nests)
    ins = _boxes([dets[i] for i in i_idx])
    i_cent = _centers(ins) if i_idx else np.zeros((0, 2))
    dist = _pair_min_dist(n_cent, i_cent)
    mpp = ctx.meters_per_pixel
    events: list[ViolationEvent] = []
    for j, ni in enumerate(n_idx):
        nest = dets[ni]
        px = float(dist[j])
        if mpp is not None and np.isfinite(px):
            meters = px * mpp
            prox = float(np.exp(-meters / 1.5))
        else:
            diag = 1.0
            if ctx.image_wh:
                diag = float(np.hypot(*ctx.image_wh))
            prox = float(np.exp(-px / max(diag * 0.08, 1.0))) if np.isfinite(px) else 0.35
        area_ratio = float(_areas(nests[j : j + 1])[0])
        if ctx.image_wh:
            area_ratio /= max(ctx.image_wh[0] * ctx.image_wh[1], 1)
        score = _clip01(nest.conf * (0.45 + 0.40 * prox + 0.15 * min(area_ratio * 80.0, 1.0)))
        events.append(
            ViolationEvent(
                kind="nest_near_insulator",
                score=score,
                message="绝缘子附近存在鸟巢隐患",
                refs=[ni] + i_idx[:1],
                extra={"pixel_dist": px if np.isfinite(px) else None, "proximity": prox},
            )
        )
    return events


@register_violation("smoke_hazard")
def eval_smoke_hazard(ctx: SceneContext) -> list[ViolationEvent]:
    events = []
    for i, d in enumerate(ctx.detections):
        if d.cls_name != "smoke":
            continue
        events.append(
            ViolationEvent(
                kind="smoke_hazard",
                score=_clip01(0.55 + 0.45 * d.conf),
                message="场景出现烟雾",
                refs=[i],
            )
        )
    return events


@register_violation("fire_hazard")
def eval_fire_hazard(ctx: SceneContext) -> list[ViolationEvent]:
    events = []
    for i, d in enumerate(ctx.detections):
        if d.cls_name != "fire":
            continue
        events.append(
            ViolationEvent(
                kind="fire_hazard",
                score=_clip01(0.80 + 0.20 * d.conf),
                message="场景出现明火",
                refs=[i],
            )
        )
    return events


@register_violation("foreign_object")
def eval_foreign_object(ctx: SceneContext) -> list[ViolationEvent]:
    events = []
    for i, d in enumerate(ctx.detections):
        if d.cls_name != "foreign_object":
            continue
        events.append(
            ViolationEvent(
                kind="foreign_object",
                score=_clip01(0.40 + 0.50 * d.conf),
                message="检测到异物侵限",
                refs=[i],
            )
        )
    return events


@register_violation("vehicle_intrusion")
def eval_vehicle_intrusion(ctx: SceneContext) -> list[ViolationEvent]:
    events = []
    for i, d in enumerate(ctx.detections):
        if d.cls_name != "vehicle":
            continue
        events.append(
            ViolationEvent(
                kind="vehicle_intrusion",
                score=_clip01(0.30 + 0.50 * d.conf),
                message="作业区域出现车辆",
                refs=[i],
            )
        )
    return events


def evaluate_violations(
    ctx: SceneContext,
    names: Iterable[str] | None = None,
) -> list[ViolationEvent]:
    selected = list(names) if names is not None else list(_REGISTRY)
    events: list[ViolationEvent] = []
    for name in selected:
        fn = _REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"unknown violation evaluator: {name}")
        events.extend(fn(ctx))
    return events


def detections_from_xyxy(
    boxes: Sequence[Sequence[float]],
    class_names: Sequence[str],
    confs: Sequence[float],
) -> list[DetectionRecord]:
    recs = []
    for box, name, conf in zip(boxes, class_names, confs):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        recs.append(DetectionRecord(cls_name=str(name), conf=float(conf), xyxy=(x1, y1, x2, y2)))
    return recs
