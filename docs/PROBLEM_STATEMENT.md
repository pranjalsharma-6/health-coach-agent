# Kaya — An Autonomous Nutrition Coach That Replans When Life Happens

> *Kaya* (काय) — Sanskrit for "body."

---

## 1. The One-Line Pitch

**Every nutrition app is a logging app. Kaya is a planning agent.**

MyFitnessPal will tell you that you ate 2,400 calories today. It will not change your
dinner because you skipped lunch. The plan stays frozen; the human is expected to adapt.

Kaya inverts that relationship. The plan is the living thing. When you skip a meal,
blow past your calorie target, sleep four hours, or eat out three nights running, an
autonomous agent notices, reasons about *why* it matters for your specific goal, and
rewrites the rest of your week — without you asking it to.

---

## 2. The Problem

### 2.1 What actually happens to people

Roughly **80% of diet plans are abandoned within the first month.** The failure mode is
almost never "the plan was nutritionally wrong." It's this:

1. A user gets a 7-day meal plan on Monday.
2. Tuesday, they skip breakfast because they're late.
3. Wednesday, a colleague brings samosas to the office.
4. By Thursday the plan no longer matches reality, so they stop opening the app.
5. By the following Monday, they've quit.

The plan didn't fail because it was bad. It failed because it was **static in a dynamic
life**, and the only mechanism available for adaptation was the user's own willpower and
nutrition knowledge — which is exactly the thing they downloaded the app to avoid needing.

### 2.2 What the tools do about it

| Category | Examples | What it does | What it doesn't do |
|---|---|---|---|
| **Loggers** | MyFitnessPal, Cronometer, Lifesum | Records what you ate against a fixed target | Never changes the plan. Adaptation is 100% on the user. |
| **Plan generators** | Eat This Much, PlateJoy | Generates a meal plan from your macros | One-shot. Regenerating is a manual button press; it has no idea you skipped Tuesday. |
| **Behavioral programs** | Noom, Second Nature | Human coaches + psychology curriculum | Expensive (₹3,000–8,000/mo), slow feedback (coach replies in hours/days), meal plans still static. |
| **Wearables** | Fitbit, Google Fit, Apple Health | Excellent passive data capture | Pure sensing. Zero planning intelligence on top of the signal. |
| **Generic LLM chat** | ChatGPT, Gemini | Great one-shot plan on request | Stateless. No memory of your adherence, no persistence, no autonomy — it only ever acts when prompted. |

**The gap is the loop.** Sensing exists (wearables). Planning exists (generators). Coaching
exists (Noom). Nobody closes the circuit so that *observed adherence automatically drives
replanning* — continuously, cheaply, and without a human in the loop.

### 2.3 The dietary-personalization gap

There is a second, narrower problem that most Western apps handle badly.

"Vegetarian" is not one thing. In India alone the meaningful categories include **pure
vegetarian, eggetarian (lacto-ovo), vegan, Jain (no root vegetables), non-vegetarian, and
halal**, layered on top of regional cuisine and religious fasting calendars. MyFitnessPal
and Eat This Much treat this as a single checkbox and then suggest quinoa bowls and turkey
sandwiches to a user who eats dal-chawal.

A plan you won't eat has an adherence rate of zero, no matter how good its macros are.
**Cultural fit is not a nice-to-have feature; it is the primary determinant of adherence**
for a large and badly-served population.

---

## 3. What We're Building

Kaya is a **full-stack, multi-user, autonomous nutrition coaching system** built on a
closed sense–evaluate–decide–act loop.

### 3.1 The core loop

```
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │   SENSE            EVALUATE           DECIDE          ACT    │
   │   ─────            ────────           ──────          ───    │
   │   meal logs   →   adherence      →   should we   →  regenerate│
   │   skips           vs. targets        replan?        remaining │
   │   weight          trend analysis     what kind      days      │
   │   steps           streak/drift       of change?     + explain │
   │   sleep                                             why       │
   │      ▲                                                 │      │
   │      └─────────────────  persist  ◄────────────────────┘      │
   └──────────────────────────────────────────────────────────────┘
```

Concretely, when a user marks lunch as skipped:

1. **Sense** — the skip event is written to the log store with a timestamp.
2. **Evaluate** — deterministic Python computes the day's remaining calorie and protein
   budget, adherence streak, and whether this is an isolated event or the third skip this
   week.
3. **Decide** — a LangGraph state machine routes: a single skip may just rebalance dinner;
   a pattern of skipped breakfasts triggers a structural replan that moves the user to a
   two-meal schedule they'll actually keep.
4. **Act** — the LLM regenerates only the affected portion of the plan, respecting the
   user's diet type, cuisine preference, allergies, and budget, and writes a
   human-readable rationale.
5. **Persist** — the new plan version is stored alongside the old one, so the user (and a
   recruiter reading the code) can see the full decision history.

### 3.2 Onboarding that actually determines the output

A short conversational intake captures the variables that drive adherence:

- **Diet type** — vegetarian / eggetarian / vegan / Jain / non-vegetarian / halal
- **Cuisine preference** — North Indian, South Indian, Continental, East Asian, mixed
- **Goal** — fat loss, muscle gain, maintenance, endurance, general health
- **Constraints** — allergies, medical flags, cooking skill, prep time available, budget tier
- **Schedule** — meals per day, typical timings, eat-out frequency

These are not stored as decoration. Every one of them is a hard constraint in the
generation prompt and a validation rule on the output.

### 3.3 Recipes, not just macros

Each meal ships with an actual **easy recipe** — ingredients, steps, prep time, and
macros grounded against a nutrition database rather than invented by the model. "300g
paneer bhurji, 24g protein" is only useful if the number is real.

---

## 4. What Makes It Innovative

These are the four things I'd defend in an interview.

### 4.1 Autonomy is architectural, not cosmetic

Most "AI agent" projects are a prompt wrapped in a `for` loop. Kaya's autonomy is a
**LangGraph state machine with conditional edges** — the graph genuinely branches based on
computed state, and different branches produce structurally different actions (no-op /
rebalance-day / structural-replan / escalate-to-user). The decision to replan is made by
the system, not by a user pressing a button.

### 4.2 Deterministic guardrails wrapping LLM judgment

Nutrition is a domain where a hallucination is a health risk. Kaya splits the work:

- **Deterministic Python owns the numbers** — BMR/TDEE via Mifflin-St Jeor, calorie floors,
  protein minimums (1.6–2.2 g/kg), maximum safe deficit. These are unit-tested functions,
  not model outputs.
- **The LLM owns judgment and language** — which foods, what phrasing, how to sequence a
  week, how to explain a change empathetically.
- **A validation layer rejects unsafe output** — any plan under 1,200 kcal, any plan
  violating the user's diet type, any plan missing the protein floor is rejected and
  regenerated. The model is never trusted to be safe on its own.

This is the single biggest differentiator from a generic LLM wrapper, and it's the part
that demonstrates actual engineering judgment.

### 4.3 Explainability as a first-class output

Every plan carries an `agent_reasoning` field, and every replan carries a diff: *what*
changed, *why* it changed, and *which observation triggered it*. The UI surfaces this as a
timeline. A user can scroll back and see "Thursday: moved your dinner protein up 20g
because you skipped lunch."

Health advice you can't interrogate is health advice you won't trust.

### 4.4 Cultural specificity as a product thesis

Diet-type modelling that treats eggetarian and Jain as first-class citizens is not a
localization checkbox — it's the wedge. It's a market that the incumbents genuinely serve
badly, and it demonstrates product thinking rather than pure technical mimicry.

---

## 5. Competitive Positioning

|  | Adapts autonomously | Real recipes | Diet-type depth | Explains itself | Cost |
|---|---|---|---|---|---|
| MyFitnessPal | ✗ | ✗ | Shallow | ✗ | Freemium |
| Eat This Much | ✗ (manual regen) | ✓ | Shallow | ✗ | ~$9/mo |
| Noom | Partial (human) | Partial | Shallow | ✓ (human) | ~$60/mo |
| Fitbit / Google Fit | ✗ | ✗ | N/A | ✗ | Free |
| ChatGPT | ✗ (stateless) | ✓ | Good if prompted | ✓ | $20/mo |
| **Kaya** | **✓ (agentic)** | **✓ (grounded)** | **✓ (6 types)** | **✓ (audit trail)** | **Free tier** |

**The defensible claim:** Kaya is the only system in this table where *the plan changes
because of what you did, without you asking it to.*

---

## 6. Why This Isn't a Generic College Project

A blunt list, because this is the question that matters for a portfolio.

| Generic project | Kaya |
|---|---|
| Single hardcoded user | Real JWT auth, multi-tenant data isolation |
| Streamlit script | FastAPI backend + Next.js frontend, independently deployable |
| LLM call in a request handler | Stateful LangGraph workflow with conditional branching |
| Trusts model output | Pydantic schema validation + domain safety rules + regeneration on failure |
| Random/mock data | Real user-logged data with a migration path to wearable OAuth |
| No tests | Unit-tested nutrition math, agent decision tests with frozen fixtures |
| "It works on my machine" | Dockerized, CI on push, health checks, structured logging |
| Spinner for 60 seconds | Streamed agent steps over SSE — the user watches it think |
| README with a screenshot | Architecture docs, decision log, evaluation methodology |

The thing a reviewer should walk away with: **this person can design a system, not just
call an API.**

---

## 7. Technical Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Next.js 14 (App Router) · TypeScript · Tailwind                   │
│  Onboarding wizard │ Plan view │ Meal logging │ Agent timeline     │
└───────────────────────────┬────────────────────────────────────────┘
                            │  REST + Server-Sent Events
┌───────────────────────────▼────────────────────────────────────────┐
│  FastAPI                                                           │
│  ┌──────────┬───────────┬────────────┬──────────────────────────┐ │
│  │  Auth    │  Profile  │   Logging  │   Agent orchestration    │ │
│  │  (JWT)   │  & prefs  │   & sense  │   (streamed)             │ │
│  └──────────┴───────────┴────────────┴───────────┬──────────────┘ │
└──────────────────────────────────────────────────┼────────────────┘
                                                   │
             ┌─────────────────────────────────────▼──────────────┐
             │  LangGraph: sense → evaluate → decide → act        │
             │  ┌────────────────────────────────────────────┐    │
             │  │  Deterministic: BMR/TDEE, budgets, safety   │    │
             │  │  LLM (Groq/Llama 3.3): generation, language │    │
             │  │  Validator: schema + domain rules + retry   │    │
             │  └────────────────────────────────────────────┘    │
             └────────────────┬───────────────────────────────────┘
                              │
             ┌────────────────▼───────────────┐
             │  MongoDB Atlas                 │
             │  users · profiles · plans      │
             │  logs · agent_events           │
             └────────────────────────────────┘
```

**Stack rationale:**

- **FastAPI** — async, native Pydantic (which the agent already speaks), auto-generated
  OpenAPI docs, and it makes the agent reusable behind an API rather than trapped in a UI.
- **Next.js** — real routing, streaming UI, and a deployment story (Vercel) that doesn't
  cold-boot for 60 seconds like Streamlit Cloud does.
- **MongoDB** — plans are deeply nested, schema-evolving documents. This is a genuine
  document-store fit, not a default choice.
- **Groq / Llama 3.3 70B** — sub-second inference at zero cost, behind a provider interface
  so swapping to GPT-4o or Claude is a config change.
- **LangGraph** — explicit, inspectable state machine. The graph *is* the documentation of
  the agent's behavior.

---

## 8. Roadmap

**Phase 1 — Foundation**
Monorepo restructure, FastAPI skeleton, auth, MongoDB models, health checks, Docker.

**Phase 2 — Agent v2**
Preference-aware planning, deterministic safety layer, validation + retry, real LangGraph
branching, agent event log.

**Phase 3 — Frontend**
Onboarding wizard, plan dashboard, meal logging with skip, streamed agent timeline,
progress charts.

**Phase 4 — The adaptive loop**
Skip/overage detection, partial-day rebalancing, structural replanning, plan diffs.

**Phase 5 — Grounding & polish**
Nutrition database integration, recipe detail, tests, CI, production deploy.

**Phase 6 — Stretch**
Google Fit OAuth, weekly email digests, voice meal logging, multi-agent
(nutritionist + trainer + critic) architecture.

---

## 9. Success Criteria

This project is done when:

1. A stranger can sign up, complete onboarding, and get a personalized plan in under 90 seconds.
2. Marking a meal skipped visibly changes tomorrow's plan, with a stated reason.
3. The agent's decision history is browsable and legible.
4. Nutrition math is unit-tested and unsafe plans are provably rejected.
5. The live link loads in under 3 seconds, cold.
6. A backend engineer reading the repo can explain the agent's behavior from the code alone.
