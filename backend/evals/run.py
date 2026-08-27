"""Agent evaluation runner.

    python -m evals.run

Reports how well the agent decides and how well the validator catches bad
plans. Deterministic — no LLM, no database, no network — so the numbers are
comparable across runs and safe to gate CI on.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List

from app.agent.graph import _choose_action
from app.agent.validators import validate_plan
from app.models.enums import AgentDecision
from evals.scenarios import (
    DECISION_SCENARIOS,
    VALIDATOR_CASES,
    build_validator_plan,
)


@dataclass
class DecisionResult:
    name: str
    situation: str
    expected: AgentDecision
    actual: AgentDecision
    detail: str
    rationale: str

    @property
    def passed(self) -> bool:
        return self.expected == self.actual


@dataclass
class ValidatorResult:
    name: str
    should_reject: bool
    did_reject: bool
    errors: List[str]
    known_gap: bool
    note: str

    @property
    def correct(self) -> bool:
        return self.should_reject == self.did_reject


@dataclass
class EvalReport:
    decisions: List[DecisionResult]
    validators: List[ValidatorResult]

    @property
    def decision_accuracy(self) -> float:
        if not self.decisions:
            return 1.0
        return sum(d.passed for d in self.decisions) / len(self.decisions)

    @property
    def scored_validators(self) -> List[ValidatorResult]:
        """Cases excluding documented gaps — the honest headline number."""
        return [v for v in self.validators if not v.known_gap]

    @property
    def validator_accuracy(self) -> float:
        scored = self.scored_validators
        if not scored:
            return 1.0
        return sum(v.correct for v in scored) / len(scored)

    @property
    def false_positives(self) -> List[ValidatorResult]:
        """Good plans wrongly rejected — the expensive failure mode.

        A false positive burns a regeneration attempt and can exhaust the retry
        budget on a plan that was fine.
        """
        return [v for v in self.validators if not v.should_reject and v.did_reject]

    @property
    def false_negatives(self) -> List[ValidatorResult]:
        """Bad plans let through — the dangerous failure mode."""
        return [
            v
            for v in self.validators
            if v.should_reject and not v.did_reject and not v.known_gap
        ]

    @property
    def known_gaps(self) -> List[ValidatorResult]:
        return [v for v in self.validators if v.known_gap]

    @property
    def passed(self) -> bool:
        return self.decision_accuracy == 1.0 and self.validator_accuracy == 1.0


def run_decision_evals() -> List[DecisionResult]:
    results = []
    for scenario in DECISION_SCENARIOS:
        state, snapshot, plan, targets = scenario.build()
        actual, detail = _choose_action(state, snapshot, plan, targets)
        results.append(
            DecisionResult(
                name=scenario.name,
                situation=scenario.situation,
                expected=scenario.expected,
                actual=actual,
                detail=detail,
                rationale=scenario.rationale,
            )
        )
    return results


def run_validator_evals() -> List[ValidatorResult]:
    results = []
    for case in VALIDATOR_CASES:
        plan, profile, targets = build_validator_plan(case)
        outcome = validate_plan(plan, profile, targets)
        results.append(
            ValidatorResult(
                name=case.name,
                should_reject=case.should_reject,
                did_reject=not outcome.is_valid,
                errors=outcome.errors,
                known_gap=case.known_gap,
                note=case.note,
            )
        )
    return results


def run() -> EvalReport:
    return EvalReport(
        decisions=run_decision_evals(), validators=run_validator_evals()
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _confusion(results: List[DecisionResult]) -> Dict[AgentDecision, Dict[AgentDecision, int]]:
    matrix: Dict[AgentDecision, Dict[AgentDecision, int]] = {
        d: {a: 0 for a in AgentDecision} for d in AgentDecision
    }
    for r in results:
        matrix[r.expected][r.actual] += 1
    return matrix


def print_report(report: EvalReport) -> None:
    print()
    print("=" * 78)
    print("  KAYA AGENT EVALUATION")
    print("=" * 78)

    # --- Decisions ---
    print("\nDECISION QUALITY")
    print("-" * 78)
    for r in report.decisions:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.name}")
        print(f"         situation: {r.situation}")
        if not r.passed:
            print(f"         expected:  {r.expected.value}")
            print(f"         actual:    {r.actual.value}")
            print(f"         agent said: {r.detail}")
            print(f"         why it matters: {r.rationale}")

    passed = sum(d.passed for d in report.decisions)
    print(
        f"\n  {passed}/{len(report.decisions)} correct "
        f"({report.decision_accuracy:.0%})"
    )

    # --- Confusion matrix ---
    print("\n  Confusion matrix (rows = expected, columns = actual)")
    labels = list(AgentDecision)
    short = {d: d.value[:9].ljust(9) for d in labels}
    header = " " * 22 + " ".join(short[d] for d in labels)
    print("  " + header)
    matrix = _confusion(report.decisions)
    for expected in labels:
        row = " ".join(
            str(matrix[expected][actual]).ljust(9) for actual in labels
        )
        print(f"  {expected.value.ljust(20)} {row}")

    # --- Validator ---
    print("\n\nVALIDATOR DETECTION")
    print("-" * 78)
    for r in report.scored_validators:
        mark = "PASS" if r.correct else "FAIL"
        verdict = "rejected" if r.did_reject else "accepted"
        want = "reject" if r.should_reject else "accept"
        print(f"  [{mark}] {r.name}: {verdict} (wanted {want})")
        if not r.correct and r.errors:
            print(f"         first error: {r.errors[0]}")

    correct = sum(v.correct for v in report.scored_validators)
    print(
        f"\n  {correct}/{len(report.scored_validators)} correct "
        f"({report.validator_accuracy:.0%})"
    )

    if report.false_positives:
        print("\n  FALSE POSITIVES (good plans rejected — burns retry budget):")
        for r in report.false_positives:
            print(f"    - {r.name}: {r.errors[0] if r.errors else '?'}")

    if report.false_negatives:
        print("\n  FALSE NEGATIVES (bad plans accepted — the dangerous kind):")
        for r in report.false_negatives:
            print(f"    - {r.name}")

    # --- Known gaps ---
    if report.known_gaps:
        print("\n\nKNOWN GAPS (excluded from the score, not from the report)")
        print("-" * 78)
        for r in report.known_gaps:
            status = "still missed" if not r.did_reject else "now caught"
            print(f"  [{status}] {r.name}")
            print(f"         {r.note}")

    print("\n" + "=" * 78)
    verdict = "PASS" if report.passed else "FAIL"
    print(
        f"  {verdict} — decisions {report.decision_accuracy:.0%}, "
        f"validator {report.validator_accuracy:.0%}"
    )
    print("=" * 78 + "\n")


def main() -> int:
    report = run()
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
