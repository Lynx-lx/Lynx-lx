"""Scene-level risk assessment: violations -> ELC thresholds -> grade."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from viscale.risk.elc import DEFAULT_LOSS_WEIGHTS, ElcResult, iterate_elc_thresholds
from viscale.risk.grading import GradeDecision, RiskGrade, decide_grade
from viscale.risk.violations import (
    DetectionRecord,
    SceneContext,
    ViolationEvent,
    evaluate_violations,
)


@dataclass
class RiskReport:
    score: float
    grade: RiskGrade
    label_zh: str
    thresholds: list[float]
    violations: list[ViolationEvent]
    elc: ElcResult | None = None
    extra: dict = field(default_factory=dict)


class RiskAssessor:
    def __init__(
        self,
        loss_weights: tuple[float, ...] = DEFAULT_LOSS_WEIGHTS,
        max_iter: int = 50,
        tol: float = 1e-4,
        violation_names: list[str] | None = None,
        initial_thresholds: tuple[float, ...] | None = None,
    ) -> None:
        self.loss_weights = tuple(loss_weights)
        self.max_iter = max_iter
        self.tol = tol
        self.violation_names = violation_names
        n_bounds = len(self.loss_weights) - 1
        if initial_thresholds is None:
            self.thresholds = np.linspace(0.25, 0.75, n_bounds)
        else:
            self.thresholds = np.asarray(initial_thresholds, dtype=np.float64)
        self._score_bank: list[float] = []
        self.last_elc: ElcResult | None = None

    def fit(self, scores: np.ndarray | list[float]) -> ElcResult:
        result = iterate_elc_thresholds(
            np.asarray(scores, dtype=np.float64),
            loss_weights=self.loss_weights,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.thresholds = result.thresholds
        self.last_elc = result
        return result

    def push_score(self, score: float) -> None:
        self._score_bank.append(float(score))

    def refit_bank(self) -> ElcResult | None:
        if len(self._score_bank) < len(self.loss_weights):
            return None
        return self.fit(self._score_bank)

    def assess(
        self,
        detections: list[DetectionRecord],
        *,
        meters_per_pixel: float | None = None,
        image_wh: tuple[int, int] | None = None,
        update_elc: bool = False,
    ) -> RiskReport:
        ctx = SceneContext(
            detections=detections,
            meters_per_pixel=meters_per_pixel,
            image_wh=image_wh,
        )
        events = evaluate_violations(ctx, names=self.violation_names)
        decision: GradeDecision = decide_grade(events, self.thresholds)
        self.push_score(decision.score)
        elc = None
        if update_elc:
            elc = self.refit_bank()
            decision = decide_grade(events, self.thresholds)
        return RiskReport(
            score=decision.score,
            grade=decision.grade,
            label_zh=decision.label_zh,
            thresholds=[float(v) for v in self.thresholds],
            violations=events,
            elc=elc or self.last_elc,
        )
