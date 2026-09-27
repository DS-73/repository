"""Deterministic underwriting score card.

The rules are *data*, not code: they live in :class:`ScoreModelConfig` which can
be loaded from a JSON file (``SS_SCORING_CONFIG_PATH``) so a score card change
does not require a deploy.  Two properties are enforced at startup:

1. **Coverage** - for every metric, the rules must partition the number line:
   no gaps (score would be undefined) and no overlaps (points double counted).
2. **Versioning** - every score carries ``model_version`` so a decision can be
   reproduced later even after the score card changes.  The version is stored on
   the application row and inside the downstream event.

Default score card (mirrors the assignment brief)::

    creditScore >= 750                        -> +30
    creditScore 700..749                      -> +20
    creditScore < 700                         ->  +5
    yearsInBusiness >= 5                      -> +20
    yearsInBusiness < 5                       -> +10
    bondAmount <= 10% of annualRevenue        -> +30
    bondAmount >  10% of annualRevenue        -> +10
    existingExposure <  20% of annualRevenue  -> +20
    existingExposure >= 20% of annualRevenue  ->  +5

    score >= 80 -> APPROVE | 50..79 -> REFER | < 50 -> DECLINE
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from suretyseven.errors import ServiceError

MAX_SCORE = 100

# --------------------------------------------------------------------- metrics
METRIC_CREDIT_SCORE = "creditScore"
METRIC_YEARS_IN_BUSINESS = "yearsInBusiness"
METRIC_BOND_TO_REVENUE = "bondToRevenueRatio"
METRIC_EXPOSURE_RATIO = "exposureToRevenueRatio"


class Operator(StrEnum):
    GE = "gte"
    GT = "gt"
    LE = "lte"
    LT = "lt"
    BETWEEN = "between"


class MetricDomain(BaseModel):
    """Valid range of a scored metric.

    Declaring the domain makes two things possible: the score card can be
    validated exhaustively (every value in the domain is scored exactly once)
    and an unknown metric name in a rule becomes a startup error instead of a
    silently ignored rule.
    """

    model_config = ConfigDict(extra="forbid")

    minimum: float = 0
    maximum: float = 1_000_000
    integer: bool = False


#: Domains for the metrics the default score card uses.
DEFAULT_METRIC_DOMAINS: dict[str, MetricDomain] = {
    METRIC_CREDIT_SCORE: MetricDomain(minimum=0, maximum=850, integer=True),
    METRIC_YEARS_IN_BUSINESS: MetricDomain(minimum=0, maximum=200, integer=True),
    # Ratios are unbounded above (an exposure can dwarf revenue); the upper
    # bound only exists to make validation finite.
    METRIC_BOND_TO_REVENUE: MetricDomain(minimum=0, maximum=1_000),
    METRIC_EXPOSURE_RATIO: MetricDomain(minimum=0, maximum=1_000),
}


class ScoringConfigError(ServiceError):
    """Raised when the score card is inconsistent (fails fast at startup)."""

    code = "SCORING_CONFIG_INVALID"
    http_status = 500


class RuleSpec(BaseModel):
    """A single add-points rule."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=64)
    metric: str = Field(min_length=1, max_length=64)
    operator: Operator
    threshold: float
    upper: float | None = None
    points: int
    description: str = ""

    @model_validator(mode="after")
    def _check_bounds(self) -> RuleSpec:
        if self.operator is Operator.BETWEEN:
            if self.upper is None:
                raise ValueError(f"rule {self.code}: 'between' requires 'upper'")
            if self.upper < self.threshold:
                raise ValueError(f"rule {self.code}: upper must be >= threshold")
        elif self.upper is not None:
            raise ValueError(f"rule {self.code}: 'upper' is only valid for 'between'")
        return self

    def matches(self, value: float) -> bool:
        match self.operator:
            case Operator.GE:
                return value >= self.threshold
            case Operator.GT:
                return value > self.threshold
            case Operator.LE:
                return value <= self.threshold
            case Operator.LT:
                return value < self.threshold
            case Operator.BETWEEN:
                return self.threshold <= value <= self._upper_bound()

    def _upper_bound(self) -> float:
        return float(self.upper) if self.upper is not None else float("inf")

    def interval(self) -> tuple[float, float]:
        """Interval representation used for overlap/gap validation."""
        match self.operator:
            case Operator.GE | Operator.GT:
                return (self.threshold, float("inf"))
            case Operator.LE | Operator.LT:
                return (float("-inf"), self.threshold)
            case Operator.BETWEEN:
                return (self.threshold, self._upper_bound())



class DecisionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve_min: int = 80
    refer_min: int = 50

    @model_validator(mode="after")
    def _ordered(self) -> DecisionThresholds:
        if self.refer_min > self.approve_min:
            raise ValueError("refer_min must be <= approve_min")
        return self


class ScoreModelConfig(BaseModel):
    """The whole score card: rules + decision thresholds + version."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1, max_length=32)
    decision_thresholds: DecisionThresholds = Field(default_factory=DecisionThresholds)
    rules: list[RuleSpec]
    metric_domains: dict[str, MetricDomain] = Field(
        default_factory=lambda: dict(DEFAULT_METRIC_DOMAINS)
    )
    #: When a ratio metric cannot be computed (annualRevenue <= 0) the model
    #: cannot score that dimension, so we never auto-approve in that case.
    escalate_when_metrics_missing: bool = True


# ------------------------------------------------------------------- defaults
def default_score_model() -> ScoreModelConfig:
    return ScoreModelConfig(
        version="2026-01-v1",
        decision_thresholds=DecisionThresholds(approve_min=80, refer_min=50),
        rules=[
            RuleSpec(
                code="CREDIT_EXCELLENT",
                metric=METRIC_CREDIT_SCORE,
                operator=Operator.GE,
                threshold=750,
                points=30,
                description="Credit score >= 750",
            ),
            RuleSpec(
                code="CREDIT_GOOD",
                metric=METRIC_CREDIT_SCORE,
                operator=Operator.BETWEEN,
                threshold=700,
                upper=749,
                points=20,
                description="Credit score 700-749",
            ),
            RuleSpec(
                code="CREDIT_WEAK",
                metric=METRIC_CREDIT_SCORE,
                operator=Operator.LT,
                threshold=700,
                points=5,
                description="Credit score < 700",
            ),
            RuleSpec(
                code="TENURE_ESTABLISHED",
                metric=METRIC_YEARS_IN_BUSINESS,
                operator=Operator.GE,
                threshold=5,
                points=20,
                description="Years in business >= 5",
            ),
            RuleSpec(
                code="TENURE_NEW",
                metric=METRIC_YEARS_IN_BUSINESS,
                operator=Operator.LT,
                threshold=5,
                points=10,
                description="Years in business < 5",
            ),
            RuleSpec(
                code="BOND_SMALL_VS_REVENUE",
                metric=METRIC_BOND_TO_REVENUE,
                operator=Operator.LE,
                threshold=0.10,
                points=30,
                description="Bond amount <= 10% of annual revenue",
            ),
            RuleSpec(
                code="BOND_LARGE_VS_REVENUE",
                metric=METRIC_BOND_TO_REVENUE,
                operator=Operator.GT,
                threshold=0.10,
                points=10,
                description="Bond amount > 10% of annual revenue",
            ),
            RuleSpec(
                code="EXPOSURE_LOW",
                metric=METRIC_EXPOSURE_RATIO,
                operator=Operator.LT,
                threshold=0.20,
                points=20,
                description="Existing exposure < 20% of annual revenue",
            ),
            RuleSpec(
                code="EXPOSURE_HIGH",
                metric=METRIC_EXPOSURE_RATIO,
                operator=Operator.GE,
                threshold=0.20,
                points=5,
                description="Existing exposure >= 20% of annual revenue",
            ),
        ],
    )


def validate_score_model(config: ScoreModelConfig) -> None:
    """Ensure every metric is scored exactly once across its whole domain.

    Integer metrics (credit score, years in business) are validated by
    enumeration, which is exact and produces a helpful error message.  Ratios are
    continuous, so their intervals are checked for gaps/overlaps instead.
    """
    by_metric: dict[str, list[RuleSpec]] = {}
    for rule in config.rules:
        if rule.metric not in config.metric_domains:
            raise ScoringConfigError(
                f"rule '{rule.code}' references unknown metric '{rule.metric}'; "
                f"known metrics: {sorted(config.metric_domains)}"
            )
        by_metric.setdefault(rule.metric, []).append(rule)

    if not by_metric:
        raise ScoringConfigError("score card must define at least one rule")

    for metric, rules in by_metric.items():
        domain = config.metric_domains[metric]
        if domain.integer:
            _validate_integer_metric(metric, rules, domain)
        else:
            _validate_continuous_metric(metric, rules, domain)


def _validate_integer_metric(
    metric: str, rules: list[RuleSpec], domain: MetricDomain
) -> None:
    for value in range(int(domain.minimum), int(domain.maximum) + 1):
        matching = [rule for rule in rules if rule.matches(float(value))]
        if not matching:
            raise ScoringConfigError(
                f"rules for metric '{metric}' leave {value} unscored "
                f"(domain {domain.minimum}-{domain.maximum})"
            )
        if len(matching) > 1:
            codes = ", ".join(rule.code for rule in matching)
            raise ScoringConfigError(
                f"rules for metric '{metric}' overlap at {value}: {codes}"
            )


def _validate_continuous_metric(
    metric: str, rules: list[RuleSpec], domain: MetricDomain
) -> None:
    intervals: list[tuple[float, float, str]] = []
    for rule in rules:
        low, high = rule.interval()
        intervals.append(
            (
                max(low, domain.minimum),
                min(high, domain.maximum),
                rule.code,
            )
        )
    intervals.sort()

    cursor = domain.minimum
    for low, high, code in intervals:
        if low > cursor:
            raise ScoringConfigError(
                f"rules for metric '{metric}' leave a gap between {cursor} and {low} "
                f"(rule: {code})"
            )
        if high < cursor:
            raise ScoringConfigError(
                f"rules for metric '{metric}' overlap below their lower bound (rule: {code})"
            )
        cursor = max(cursor, high)
    if cursor < domain.maximum:
        raise ScoringConfigError(
            f"rules for metric '{metric}' do not cover values above {cursor}"
        )


def load_score_model(path: str | Path | None = None) -> ScoreModelConfig:
    """Load the score card from JSON, or return the built-in default."""
    if path is None:
        config = default_score_model()
    else:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        config = ScoreModelConfig.model_validate(raw)
    validate_score_model(config)
    return config


# -------------------------------------------------------------------- results
@dataclass(slots=True)
class ScoreResult:
    score: int
    decision: str
    model_version: str
    breakdown: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "decision": self.decision,
            "modelVersion": self.model_version,
            "breakdown": self.breakdown,
            "reasons": self.reasons,
            "missingMetrics": self.missing_metrics,
        }


# ------------------------------------------------------------------ computing
def compute_metrics(
    *,
    bond_amount: Decimal,
    annual_revenue: Decimal,
    years_in_business: int,
    credit_score: int,
    existing_exposure: Decimal,
) -> tuple[dict[str, float | None], list[str]]:
    """Derive the metrics the score card operates on.

    Ratio metrics are ``None`` when ``annual_revenue <= 0`` because dividing by
    zero is undefined; the caller decides what to do (we escalate to REFER).
    """
    metrics: dict[str, float | None] = {
        METRIC_CREDIT_SCORE: float(credit_score),
        METRIC_YEARS_IN_BUSINESS: float(years_in_business),
    }
    missing: list[str] = []
    if annual_revenue and annual_revenue > 0:
        revenue = Decimal(annual_revenue)
        metrics[METRIC_BOND_TO_REVENUE] = float(bond_amount / revenue)
        metrics[METRIC_EXPOSURE_RATIO] = float(existing_exposure / revenue)
    else:
        metrics[METRIC_BOND_TO_REVENUE] = None
        metrics[METRIC_EXPOSURE_RATIO] = None
        missing = [METRIC_BOND_TO_REVENUE, METRIC_EXPOSURE_RATIO]
    return metrics, missing


def decide(config: ScoreModelConfig, score: int, *, escalate: bool) -> tuple[str, list[str]]:
    """Map a score to a decision, honouring the missing-metric escalation."""
    thresholds = config.decision_thresholds
    if score >= thresholds.approve_min:
        decision = "APPROVE"
        reason = f"score {score} >= approve threshold {thresholds.approve_min}"
    elif score >= thresholds.refer_min:
        decision = "REFER"
        reason = (
            f"score {score} inside refer band "
            f"{thresholds.refer_min}-{thresholds.approve_min - 1}"
        )
    else:
        decision = "DECLINE"
        reason = f"score {score} < refer threshold {thresholds.refer_min}"

    reasons = [reason]
    if escalate and decision == "APPROVE":
        decision = "REFER"
        reasons.append(
            "escalated to REFER: at least one metric could not be computed from the "
            "applicant data (see missingMetrics)"
        )
    return decision, reasons


def evaluate(
    config: ScoreModelConfig,
    *,
    bond_amount: Decimal,
    annual_revenue: Decimal,
    years_in_business: int,
    credit_score: int,
    existing_exposure: Decimal,
) -> ScoreResult:
    """Run the score card and return the score, breakdown and decision."""
    metrics, missing = compute_metrics(
        bond_amount=bond_amount,
        annual_revenue=annual_revenue,
        years_in_business=years_in_business,
        credit_score=credit_score,
        existing_exposure=existing_exposure,
    )

    score = 0
    breakdown: list[dict[str, Any]] = []
    for rule in config.rules:
        observed = metrics.get(rule.metric)
        if observed is None:
            breakdown.append(
                {
                    "rule": rule.code,
                    "metric": rule.metric,
                    "description": rule.description,
                    "observed": None,
                    "points": 0,
                    "applied": False,
                    "note": "metric unavailable - rule not evaluated",
                }
            )
            continue
        applied = rule.matches(observed)
        points = rule.points if applied else 0
        score += points
        breakdown.append(
            {
                "rule": rule.code,
                "metric": rule.metric,
                "description": rule.description,
                "observed": observed,
                "points": points,
                "applied": applied,
            }
        )

    score = max(0, min(MAX_SCORE, score))
    escalate = bool(missing) and config.escalate_when_metrics_missing
    decision, reasons = decide(config, score, escalate=escalate)
    return ScoreResult(
        score=score,
        decision=decision,
        model_version=config.version,
        breakdown=breakdown,
        reasons=reasons,
        missing_metrics=missing,
    )

    return config
