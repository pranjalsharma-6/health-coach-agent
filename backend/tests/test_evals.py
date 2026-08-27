"""Runs the eval suite as a test so CI catches regressions in agent quality.

The suite is also runnable on its own for the full report:

    python -m evals.run
"""

from evals.run import run


def test_agent_decides_correctly_in_every_scenario():
    report = run()

    failures = [
        f"{d.name}: expected {d.expected.value}, got {d.actual.value} "
        f"({d.situation})"
        for d in report.decisions
        if not d.passed
    ]
    assert not failures, "Agent made the wrong call:\n  " + "\n  ".join(failures)


def test_validator_has_no_false_negatives():
    """A bad plan getting through is the failure mode that matters."""
    report = run()

    missed = [v.name for v in report.false_negatives]
    assert not missed, (
        "Validator accepted plans it should have rejected: " + ", ".join(missed)
    )


def test_validator_has_no_false_positives():
    """Rejecting a good plan burns a regeneration attempt for nothing."""
    report = run()

    wrong = [
        f"{v.name} ({v.errors[0] if v.errors else 'no error given'})"
        for v in report.false_positives
    ]
    assert not wrong, (
        "Validator rejected plans it should have accepted:\n  " + "\n  ".join(wrong)
    )


def test_known_gaps_have_not_silently_changed():
    """Pin the documented misses.

    If one starts being caught, that's good news — but the note claiming it's a
    gap is then stale, so the suite should say so rather than drift.
    """
    report = run()

    unexpectedly_caught = [v.name for v in report.known_gaps if v.did_reject]
    assert not unexpectedly_caught, (
        "These are recorded as known gaps but are now detected — update the "
        "notes in evals/scenarios.py: " + ", ".join(unexpectedly_caught)
    )


def test_overall_suite_passes():
    report = run()
    assert report.passed, (
        f"decisions {report.decision_accuracy:.0%}, "
        f"validator {report.validator_accuracy:.0%}"
    )
