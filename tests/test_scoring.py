"""Score card tests: rules, boundaries, decisions and configuration guards."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from suretyseven.scoring import (
    METRIC_BOND_TO_REVENUE,
    METRIC_CREDIT_SCORE,
    METRIC_EXPOSURE_RATIO,
    DecisionThresholds,
    MetricDomain,
    Operator,
    RuleSpec,
    ScoreModelConfig,
    ScoreResult,
    ScoringConfigError,
    default_score_model,
    evaluate,
    load_score_model,
    validate_score_model,
)


def score(**overrides) -> ScoreResult:
    params: dict = {
        "bond_amount": Decimal("500000"),
        "annual_revenue": Decimal("12000000"),
        "years_in_business": 8,
        "credit_score": 760,
        "existing_exposure": Decimal("1500000"),
    }
    params.update(overrides)
    return evaluate(default_score_model(), **params)


def points_for(result: ScoreResult, *rules: str) -> int:
    breakdown = {item["rule"]: item["points"] for item in result.breakdown}
    return sum(breakdown[rule] for rule in rules)


@pytest.mark.parametrize(
    ("credit_score", "expected"),
    [(850, 30), (750, 30), (749, 20), (700, 20), (699, 5), (500, 5), (0, 5)],
)
def test_credit_score_bands(credit_score: int, expected: int) -> None:
    result = score(credit_score=credit_score)
    assert points_for(result, "CREDIT_EXCELLENT", "CREDIT_GOOD", "CREDIT_WEAK") == expected


@pytest.mark.parametrize(("years", "expected"), [(0, 10), (4, 10), (5, 20), (50, 20)])
def test_years_in_business_bands(years: int, expected: int) -> None:
    result = score(years_in_business=years)
    assert points_for(result, "TENURE_ESTABLISHED", "TENURE_NEW") == expected


@pytest.mark.parametrize(
    ("bond_amount", "expected"),
    [
        (Decimal("1199999"), 30),
        (Decimal("1200000"), 30),
        (Decimal("1200001"), 10),
        (Decimal("10000000"), 10),
    ],
)
def test_bond_to_revenue_ratio_bands(bond_amount: Decimal, expected: int) -> None:
    result = score(bond_amount=bond_amount)
    assert points_for(result, "BOND_SMALL_VS_REVENUE", "BOND_LARGE_VS_REVENUE") == expected


@pytest.mark.parametrize(
    ("exposure", "expected"), [(Decimal("2399999"), 20), (Decimal("2400000"), 5)]
)
def test_exposure_ratio_bands(exposure: Decimal, expected: int) -> None:
    result = score(existing_exposure=exposure)
    assert points_for(result, "EXPOSURE_LOW", "EXPOSURE_HIGH") == expected


def test_best_case_is_approved_with_full_breakdown() -> None:
    result = score()
    assert result.score == 100
    assert result.decision == "APPROVE"
    assert result.model_version == "2026-01-v1"
    assert sum(item["points"] for item in result.breakdown) == result.score
    assert len(result.breakdown) == len(default_score_model().rules)


def test_worst_case_is_declined() -> None:
    result = score(
        bond_amount=Decimal("1000000"),
        annual_revenue=Decimal("1000000"),
        years_in_business=0,
        credit_score=500,
        existing_exposure=Decimal("900000"),
    )
    assert result.score == 30
    assert result.decision == "DECLINE"
    assert any("refer threshold" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("flat_score", "expected"),
    [(80, "APPROVE"), (79, "REFER"), (50, "REFER"), (49, "DECLINE")],
)
def test_decision_thresholds(flat_score: int, expected: str) -> None:
    """Decision boundaries are configuration driven, not hard coded."""
    model = ScoreModelConfig(
        version="flat",
        decision_thresholds=DecisionThresholds(approve_min=80, refer_min=50),
        rules=[
            RuleSpec(
                code="FLAT",
                metric=METRIC_CREDIT_SCORE,
                operator=Operator.GE,
                threshold=0,
                points=flat_score,
                description="flat points equal to the score under test",
            )
        ],
        metric_domains={
            METRIC_CREDIT_SCORE: MetricDomain(minimum=0, maximum=850, integer=True)
        },
    )
    validate_score_model(model)
    result = evaluate(
        model,
        bond_amount=Decimal("1"),
        annual_revenue=Decimal("1"),
        years_in_business=1,
        credit_score=1,
        existing_exposure=Decimal("0"),
    )
    assert result.decision == expected


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        DecisionThresholds(approve_min=50, refer_min=80)


def test_missing_revenue_escalates_approve_to_refer() -> None:
    """A would-be approval with unscored metrics must go to a human, never auto-approve."""
    model = ScoreModelConfig(
        version="escalation-probe",
        decision_thresholds=DecisionThresholds(approve_min=40, refer_min=20),
        rules=list(default_score_model().rules),
    )
    result = evaluate(
        model,
        bond_amount=Decimal("500000"),
        annual_revenue=Decimal("0"),
        years_in_business=8,
        credit_score=760,
        existing_exposure=Decimal("1500000"),
    )
    assert set(result.missing_metrics) == {METRIC_BOND_TO_REVENUE, METRIC_EXPOSURE_RATIO}
    assert result.score == 50  # credit + tenure rules only
    assert result.decision == "REFER"  # would have been APPROVE at 50 points
    assert any("escalated to REFER" in reason for reason in result.reasons)
    unevaluated = [item for item in result.breakdown if item["observed"] is None]
    assert len(unevaluated) == 4
    assert all(item["applied"] is False for item in unevaluated)


def test_score_card_can_be_overridden_from_json(tmp_path: Path) -> None:
    config = {
        "version": "custom-v9",
        "decision_thresholds": {"approve_min": 100, "refer_min": 10},
        "rules": [
            {
                "code": "CREDIT_ONLY",
                "metric": "creditScore",
                "operator": "gte",
                "threshold": 700,
                "points": 100,
                "description": "Credit score >= 700",
            },
            {
                "code": "CREDIT_FLOOR",
                "metric": "creditScore",
                "operator": "lt",
                "threshold": 700,
                "points": 10,
                "description": "Credit score < 700",
            },
        ],
    }
    path = tmp_path / "score-card.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    model = load_score_model(path)
    assert model.version == "custom-v9"

    def run(credit_score: int) -> ScoreResult:
        return evaluate(
            model,
            bond_amount=Decimal("1"),
            annual_revenue=Decimal("1"),
            years_in_business=1,
            credit_score=credit_score,
            existing_exposure=Decimal("0"),
        )

    assert (run(700).decision, run(700).score) == ("APPROVE", 100)
    assert (run(699).decision, run(699).score) == ("REFER", 10)


def test_unknown_metric_is_rejected() -> None:
    model = default_score_model()
    model.rules[0].metric = "madeUpMetric"
    with pytest.raises(ScoringConfigError, match="unknown metric"):
        validate_score_model(model)


def test_gap_in_integer_rules_is_rejected() -> None:
    model = ScoreModelConfig(
        version="broken",
        rules=[
            RuleSpec(
                code="HIGH_ONLY",
                metric=METRIC_CREDIT_SCORE,
                operator=Operator.GE,
                threshold=750,
                points=30,
            )
        ],
    )
    with pytest.raises(ScoringConfigError, match="unscored"):
        validate_score_model(model)


def test_overlapping_rules_are_rejected() -> None:
    model = ScoreModelConfig(
        version="broken",
        rules=[
            RuleSpec(
                code="COVERS_EVERYTHING",
                metric=METRIC_CREDIT_SCORE,
                operator=Operator.GE,
                threshold=0,
                points=30,
            ),
            RuleSpec(
                code="ALSO_MATCHES",
                metric=METRIC_CREDIT_SCORE,
                operator=Operator.GE,
                threshold=700,
                points=20,
            ),
        ],
    )
    with pytest.raises(ScoringConfigError, match="overlap"):
        validate_score_model(model)


def test_gap_in_continuous_rules_is_rejected() -> None:
    model = ScoreModelConfig(
        version="broken",
        rules=[
            RuleSpec(
                code="SMALL",
                metric=METRIC_BOND_TO_REVENUE,
                operator=Operator.LE,
                threshold=0.1,
                points=30,
            ),
            RuleSpec(
                code="VERY_LARGE",
                metric=METRIC_BOND_TO_REVENUE,
                operator=Operator.GT,
                threshold=0.5,
                points=10,
            ),
        ],
    )
    with pytest.raises(ScoringConfigError, match="gap"):
        validate_score_model(model)


def test_between_rule_requires_upper_bound() -> None:
    with pytest.raises(ValueError, match="requires 'upper'"):
        RuleSpec(
            code="BROKEN",
            metric=METRIC_CREDIT_SCORE,
            operator=Operator.BETWEEN,
            threshold=700,
            points=10,
        )

