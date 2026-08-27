# Agent evaluation suite

```bash
cd backend && python -m evals.run
```

A test asserts that code does what it does. An **eval** measures whether the
agent *decides well* — and records the cases it gets wrong instead of quietly
leaving them out.

Everything here is deterministic. The decision rules are plain Python and the
validator is plain Python, so the suite needs no LLM, no database and no
network. The same numbers come out on every run, which is what makes them worth
gating CI on.

---

## What it measures

### 1. Decision quality — 13 scenarios

Each scenario is a situation the agent will actually meet, paired with the call
it should make and *why that call matters*:

| Scenario | Expected | Why |
|---|---|---|
| `cold_start` | create initial | Nothing to adapt yet |
| `on_track` | no action | Intervening when nothing is wrong erodes trust |
| `never_logged` | no action | Silence is missing data, not non-adherence |
| `single_skip_recoverable` | rebalance day | The day can still be salvaged |
| `single_skip_day_over` | no action | No remaining meal to rebalance into |
| `three_day_skip_streak` | structural replan | A pattern means the plan is wrong |
| `low_adherence_sufficient_evidence` | structural replan | Enough evidence to rebuild |
| `low_adherence_insufficient_evidence` | rebalance day | 50% of n=2 proves nothing |
| `plan_expired` | structural replan | The block is finished |
| `calorie_overage_recoverable` | rebalance day | Lighten what's left |
| `calorie_overage_day_over` | no action | Nothing left to adjust |
| `user_forced_rebuild` | structural replan | An explicit request outranks judgement |
| `severity_ordering` | structural replan | The streak must beat the day rebalance |

The report prints a **confusion matrix**, so a systematic bias — say, reaching
for a structural replan when a day rebalance would do — shows up as an
off-diagonal cluster rather than a single failing assertion.

### 2. Validator detection — 17 scored cases

Deliberately broken plans, checking the safety layer actually catches them:
diet violations across all six diet types, allergens hidden in descriptions,
protein below the floor, calories outside tolerance, macros that don't reconcile
with their own calorie count, structural faults, and one case that must *not*
fire (`hamper` must not trip the `ham` keyword).

Two failure modes are reported separately because they cost different things:

- **False negatives** — a bad plan gets through. The dangerous one.
- **False positives** — a good plan is rejected. Burns a regeneration attempt
  and can exhaust the retry budget on a plan that was fine.

---

## Known gaps

Two cases are recorded as **known misses** and excluded from the score. They are
in the suite precisely so the number stays honest — a detector evaluated only on
cases it was built to catch will always report 100%.

| Gap | Why it's missed |
|---|---|
| `compound_word_dairy_for_vegan` | "milk" inside "milkshake". The scan matches whole words plus a plural suffix; tightening it to catch compounds reintroduces the `ham`/`hamper` false positive. The prompt constraint is the primary defence here. |
| `brand_name_hides_meat` | A keyword scan cannot know a brand name implies chicken. Catching it needs ingredient-level lookup, not a word list. |

`test_known_gaps_have_not_silently_changed` pins them: if one starts being
caught, the test fails so the note gets updated rather than drifting out of date.

---

## Adding a scenario

Append to `DECISION_SCENARIOS` in `scenarios.py`:

```python
DecisionScenario(
    name="illness_pause",
    situation="User logged three days of low steps and poor sleep",
    expected=AgentDecision.STRUCTURAL_REPLAN,
    build=_illness_pause,
    rationale="Training volume should drop when recovery signals do.",
)
```

`build` returns the `(state, snapshot, plan, targets)` tuple that
`_choose_action` takes. Use the helpers in `tests/factories.py` rather than
constructing models by hand.

If the new scenario fails, that's the point — either the rule needs fixing or
the expectation was wrong. Decide which before changing anything.

---

## Related

- `tests/test_properties.py` — property-based tests proving the nutrition
  engine's safety invariants hold across the *entire* valid input space, not
  just chosen examples. Hypothesis found the small-sedentary-user case where
  the calorie floor pushes the target above maintenance.
- `tests/test_evals.py` — runs this suite under pytest so CI catches
  regressions.
