"""Risk fusion and four-level grading for power-security scenes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from viscale.risk.violations import ViolationEvent


class RiskGrade(IntEnum):
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


GRADE_LABELS_ZH = {
    RiskGrade.LOW: "低风险",
    RiskGrade.MODERATE: "一般风险",
    RiskGrade.HIGH: "较大风险",
    RiskGrade.CRITICAL: "重大风险",
}

HARD_FLOOR: dict[str, float] = {
    "fire_hazard": 0.85,
    "smoke_hazard": 0.60,
}


def fuse_scores(events: list[ViolationEvent]) -> float:
    """Noisy-OR fusion of per-violation scores in [0, 1]."""
    if not events:
        return 0.0
    survival = 1.0
    for ev in events:
        survival *= 1.0 - float(np.clip(ev.score, 0.0, 1.0))
    fused = 1.0 - survival
    floors = [HARD_FLOOR[ev.kind] for ev in events if ev.kind in HARD_FLOOR]
    if floors:
        fused = max(fused, max(floors))
    return float(np.clip(fused, 0.0, 1.0))


def grade_from_thresholds(score: float, thresholds: np.ndarray) -> RiskGrade:
    """Map a scalar score to K=len(thresholds)+1 ordered grades."""
    idx = int(np.searchsorted(np.asarray(thresholds, dtype=np.float64), score, side="right"))
    n_grades = int(np.asarray(thresholds).size) + 1
    idx = min(max(idx, 0), n_grades - 1)
    mapping = (RiskGrade.LOW, RiskGrade.MODERATE, RiskGrade.HIGH, RiskGrade.CRITICAL)
    if n_grades == 4:
        return mapping[idx]
    return RiskGrade(min(idx + 1, int(RiskGrade.CRITICAL)))


def apply_rule_overrides(events: list[ViolationEvent], grade: RiskGrade) -> RiskGrade:
    kinds = {ev.kind for ev in events}
    if "fire_hazard" in kinds and grade < RiskGrade.CRITICAL:
        return RiskGrade.CRITICAL
    if "smoke_hazard" in kinds and grade < RiskGrade.HIGH:
        return RiskGrade.HIGH
    return grade


@dataclass
class GradeDecision:
    score: float
    grade: RiskGrade
    label_zh: str
    thresholds: np.ndarray


def decide_grade(
    events: list[ViolationEvent],
    thresholds: np.ndarray,
) -> GradeDecision:
    score = fuse_scores(events)
    grade = apply_rule_overrides(events, grade_from_thresholds(score, thresholds))
    return GradeDecision(
        score=score,
        grade=grade,
        label_zh=GRADE_LABELS_ZH.get(grade, grade.name),
        thresholds=np.asarray(thresholds, dtype=np.float64),
    )
