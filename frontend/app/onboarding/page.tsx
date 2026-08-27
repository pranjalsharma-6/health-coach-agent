"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { Alert, Button, ChoiceCard, Field, Input, cn } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { useRequireAuth, useAuth } from "@/lib/auth";
import {
  ACTIVITY_DESCRIPTIONS,
  ACTIVITY_LABELS,
  BUDGET_LABELS,
  CUISINE_LABELS,
  DIET_DESCRIPTIONS,
  DIET_EMOJI,
  DIET_LABELS,
  GOAL_DESCRIPTIONS,
  GOAL_EMOJI,
  GOAL_LABELS,
  SKILL_DESCRIPTIONS,
  SKILL_LABELS,
} from "@/lib/labels";
import type {
  ActivityLevel,
  BudgetTier,
  CookingSkill,
  Cuisine,
  DietType,
  Gender,
  Goal,
  ProfileDraft,
} from "@/lib/types";

const STEPS = [
  "Your goal",
  "About you",
  "How active",
  "How you eat",
  "Your kitchen",
  "Review",
] as const;

const DEFAULT_DRAFT: ProfileDraft = {
  gender: "male",
  age_years: 25,
  height_cm: 170,
  current_weight_kg: 70,
  target_weight_kg: 65,
  goal: "fat_loss",
  activity_level: "moderately_active",
  target_timeline_weeks: 12,
  diet_type: "vegetarian",
  cuisine_preferences: ["north_indian"],
  allergies: [],
  disliked_foods: [],
  meals_per_day: 4,
  cooking_skill: "beginner",
  max_prep_minutes: 30,
  budget_tier: "medium",
  eat_out_per_week: 2,
  medical_notes: null,
};

export default function OnboardingPage() {
  const { loading } = useRequireAuth({ requireOnboarded: false });
  const { refresh } = useAuth();
  const router = useRouter();

  const [step, setStep] = useState(0);
  const [draft, setDraft] = useState<ProfileDraft>(DEFAULT_DRAFT);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function update<K extends keyof ProfileDraft>(key: K, value: ProfileDraft[K]) {
    setDraft((prev) => ({ ...prev, [key]: value }));
  }

  /** Add or remove one cuisine, keeping at least one selected.

   * Deselecting the last one would leave the planner with no cuisine guidance
   * at all, so it falls back to "Mixed" — which is what an empty selection
   * means anyway. Choosing anything else then clears Mixed, since "mixed plus
   * South Indian" is not a coherent instruction.
   */
  function toggleCuisine(cuisine: Cuisine) {
    setDraft((prev) => {
      const selected = prev.cuisine_preferences;

      if (selected.includes(cuisine)) {
        const remaining = selected.filter((c) => c !== cuisine);
        return {
          ...prev,
          cuisine_preferences: remaining.length ? remaining : ["mixed"],
        };
      }

      if (cuisine === "mixed") {
        return { ...prev, cuisine_preferences: ["mixed"] };
      }

      return {
        ...prev,
        cuisine_preferences: [
          ...selected.filter((c) => c !== "mixed"),
          cuisine,
        ],
      };
    });
  }

  // Target weight is only meaningful for weight-change goals.
  const needsTargetWeight =
    draft.goal === "fat_loss" || draft.goal === "muscle_gain";

  const stepValid = useMemo(() => {
    switch (step) {
      case 1:
        return (
          draft.age_years >= 13 &&
          draft.age_years <= 100 &&
          draft.height_cm >= 100 &&
          draft.height_cm <= 250 &&
          draft.current_weight_kg >= 30 &&
          draft.current_weight_kg <= 300 &&
          (!needsTargetWeight ||
            (draft.target_weight_kg !== null &&
              draft.target_weight_kg >= 30 &&
              draft.target_weight_kg <= 300))
        );
      default:
        return true;
    }
  }, [step, draft, needsTargetWeight]);

  async function handleSubmit() {
    setError(null);
    setSubmitting(true);
    try {
      await api.profile.create({
        ...draft,
        target_weight_kg: needsTargetWeight ? draft.target_weight_kg : null,
      });
      await refresh();
      router.push("/dashboard?welcome=1");
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Couldn't save your profile. Please try again.",
      );
      setSubmitting(false);
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen grid place-items-center">
        <div className="skeleton size-12 rounded-full" />
      </div>
    );
  }

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="max-w-2xl mx-auto">
        <StepIndicator current={step} />

        <div key={step} className="animate-fade-up">
          {step === 0 && (
            <StepShell
              title="What brings you here?"
              subtitle="This shapes your calorie target and how your week is structured."
            >
              <div className="grid sm:grid-cols-2 gap-3">
                {(Object.keys(GOAL_LABELS) as Goal[]).map((goal) => (
                  <ChoiceCard
                    key={goal}
                    selected={draft.goal === goal}
                    onSelect={() => update("goal", goal)}
                    emoji={GOAL_EMOJI[goal]}
                    title={GOAL_LABELS[goal]}
                    description={GOAL_DESCRIPTIONS[goal]}
                  />
                ))}
              </div>
            </StepShell>
          )}

          {step === 1 && (
            <StepShell
              title="A bit about you"
              subtitle="Your calorie and protein targets are calculated from these — nothing here is guessed."
            >
              <div className="space-y-5">
                <Field label="Gender">
                  <div className="grid grid-cols-3 gap-2">
                    {(["male", "female", "other"] as Gender[]).map((g) => (
                      <ChoiceCard
                        key={g}
                        selected={draft.gender === g}
                        onSelect={() => update("gender", g)}
                        title={g === "other" ? "Other" : g === "male" ? "Male" : "Female"}
                      />
                    ))}
                  </div>
                </Field>

                <div className="grid sm:grid-cols-3 gap-4">
                  <Field label="Age" hint="years" htmlFor="age">
                    <Input
                      id="age"
                      type="number"
                      min={13}
                      max={100}
                      value={draft.age_years}
                      onChange={(e) =>
                        update("age_years", Number(e.target.value))
                      }
                    />
                  </Field>
                  <Field label="Height" hint="cm" htmlFor="height">
                    <Input
                      id="height"
                      type="number"
                      min={100}
                      max={250}
                      value={draft.height_cm}
                      onChange={(e) =>
                        update("height_cm", Number(e.target.value))
                      }
                    />
                  </Field>
                  <Field label="Weight" hint="kg" htmlFor="weight">
                    <Input
                      id="weight"
                      type="number"
                      step="0.1"
                      min={30}
                      max={300}
                      value={draft.current_weight_kg}
                      onChange={(e) =>
                        update("current_weight_kg", Number(e.target.value))
                      }
                    />
                  </Field>
                </div>

                {needsTargetWeight && (
                  <div className="grid sm:grid-cols-2 gap-4">
                    <Field label="Target weight" hint="kg" htmlFor="target">
                      <Input
                        id="target"
                        type="number"
                        step="0.1"
                        min={30}
                        max={300}
                        value={draft.target_weight_kg ?? ""}
                        onChange={(e) =>
                          update("target_weight_kg", Number(e.target.value))
                        }
                      />
                    </Field>
                    <Field
                      label="Over how long?"
                      hint="weeks — Kaya caps the rate if this is too fast"
                      htmlFor="timeline"
                    >
                      <Input
                        id="timeline"
                        type="number"
                        min={4}
                        max={52}
                        value={draft.target_timeline_weeks}
                        onChange={(e) =>
                          update(
                            "target_timeline_weeks",
                            Number(e.target.value),
                          )
                        }
                      />
                    </Field>
                  </div>
                )}

                {!stepValid && (
                  <Alert tone="warning">
                    Please check these values — something&apos;s outside a
                    realistic range.
                  </Alert>
                )}
              </div>
            </StepShell>
          )}

          {step === 2 && (
            <StepShell
              title="How active are you?"
              subtitle="Be honest rather than aspirational — an inflated answer means an inflated calorie target."
            >
              <div className="space-y-3">
                {(Object.keys(ACTIVITY_LABELS) as ActivityLevel[]).map(
                  (level) => (
                    <ChoiceCard
                      key={level}
                      selected={draft.activity_level === level}
                      onSelect={() => update("activity_level", level)}
                      title={ACTIVITY_LABELS[level]}
                      description={ACTIVITY_DESCRIPTIONS[level]}
                    />
                  ),
                )}
              </div>
            </StepShell>
          )}

          {step === 3 && (
            <StepShell
              title="How do you eat?"
              subtitle="This is the most important answer here. A plan you won't eat is worth nothing, however good its macros are."
            >
              <div className="space-y-6">
                <Field label="Diet">
                  <div className="grid sm:grid-cols-2 gap-3">
                    {(Object.keys(DIET_LABELS) as DietType[]).map((diet) => (
                      <ChoiceCard
                        key={diet}
                        selected={draft.diet_type === diet}
                        onSelect={() => update("diet_type", diet)}
                        emoji={DIET_EMOJI[diet]}
                        title={DIET_LABELS[diet]}
                        description={DIET_DESCRIPTIONS[diet]}
                      />
                    ))}
                  </div>
                </Field>

                <Field
                  label="Food you actually cook and enjoy"
                  hint="Pick as many as you like — Kaya draws from all of them across the week."
                >
                  <div className="grid sm:grid-cols-3 gap-2">
                    {(Object.keys(CUISINE_LABELS) as Cuisine[]).map((c) => (
                      <ChoiceCard
                        key={c}
                        selected={draft.cuisine_preferences.includes(c)}
                        onSelect={() => toggleCuisine(c)}
                        title={CUISINE_LABELS[c]}
                      />
                    ))}
                  </div>
                </Field>

                <TagInput
                  label="Allergies"
                  hint="Kaya will never suggest these. Press Enter after each."
                  values={draft.allergies}
                  onChange={(v) => update("allergies", v)}
                  placeholder="peanut, shellfish…"
                />

                <TagInput
                  label="Foods you'd rather avoid"
                  hint="Not allergies — just things you dislike."
                  values={draft.disliked_foods}
                  onChange={(v) => update("disliked_foods", v)}
                  placeholder="mushroom, bitter gourd…"
                />
              </div>
            </StepShell>
          )}

          {step === 4 && (
            <StepShell
              title="Your kitchen, realistically"
              subtitle="Kaya plans for the time and budget you actually have, not the ones you wish you had."
            >
              <div className="space-y-6">
                <Field label="Meals a day">
                  <div className="grid grid-cols-5 gap-2">
                    {[2, 3, 4, 5, 6].map((n) => (
                      <ChoiceCard
                        key={n}
                        selected={draft.meals_per_day === n}
                        onSelect={() => update("meals_per_day", n)}
                        title={String(n)}
                        className="text-center"
                      />
                    ))}
                  </div>
                </Field>

                <Field label="How comfortable are you cooking?">
                  <div className="grid sm:grid-cols-3 gap-2">
                    {(Object.keys(SKILL_LABELS) as CookingSkill[]).map((s) => (
                      <ChoiceCard
                        key={s}
                        selected={draft.cooking_skill === s}
                        onSelect={() => update("cooking_skill", s)}
                        title={SKILL_LABELS[s]}
                        description={SKILL_DESCRIPTIONS[s]}
                      />
                    ))}
                  </div>
                </Field>

                <div className="grid sm:grid-cols-2 gap-4">
                  <Field
                    label={`Max prep time: ${draft.max_prep_minutes} min`}
                    hint="per meal"
                  >
                    <input
                      type="range"
                      min={5}
                      max={90}
                      step={5}
                      value={draft.max_prep_minutes}
                      onChange={(e) =>
                        update("max_prep_minutes", Number(e.target.value))
                      }
                      className="w-full accent-brand-600"
                    />
                  </Field>

                  <Field
                    label={`Eating out: ${draft.eat_out_per_week}x a week`}
                    hint="Kaya leaves room for these"
                  >
                    <input
                      type="range"
                      min={0}
                      max={14}
                      value={draft.eat_out_per_week}
                      onChange={(e) =>
                        update("eat_out_per_week", Number(e.target.value))
                      }
                      className="w-full accent-brand-600"
                    />
                  </Field>
                </div>

                <Field label="Grocery budget">
                  <div className="grid grid-cols-3 gap-2">
                    {(Object.keys(BUDGET_LABELS) as BudgetTier[]).map((b) => (
                      <ChoiceCard
                        key={b}
                        selected={draft.budget_tier === b}
                        onSelect={() => update("budget_tier", b)}
                        title={BUDGET_LABELS[b]}
                      />
                    ))}
                  </div>
                </Field>

                <Field
                  label="Anything medical Kaya should know?"
                  hint="Optional. Kaya is not a doctor and will suggest seeing one where it matters."
                  htmlFor="medical"
                >
                  <Input
                    id="medical"
                    value={draft.medical_notes ?? ""}
                    onChange={(e) =>
                      update("medical_notes", e.target.value || null)
                    }
                    placeholder="e.g. PCOS, pre-diabetic, lactose intolerant"
                  />
                </Field>
              </div>
            </StepShell>
          )}

          {step === 5 && (
            <StepShell
              title="Does this look right?"
              subtitle="Everything here becomes a hard constraint on your plan. You can change any of it later."
            >
              <div className="space-y-3">
                {error && <Alert tone="error">{error}</Alert>}

                <ReviewRow label="Goal" value={GOAL_LABELS[draft.goal]} />
                <ReviewRow
                  label="You"
                  value={`${draft.age_years}y · ${draft.height_cm}cm · ${draft.current_weight_kg}kg`}
                />
                {needsTargetWeight && draft.target_weight_kg && (
                  <ReviewRow
                    label="Target"
                    value={`${draft.target_weight_kg}kg in ${draft.target_timeline_weeks} weeks`}
                  />
                )}
                <ReviewRow
                  label="Activity"
                  value={ACTIVITY_LABELS[draft.activity_level]}
                />
                <ReviewRow
                  label="Diet"
                  value={`${DIET_EMOJI[draft.diet_type]} ${DIET_LABELS[draft.diet_type]} · ${draft.cuisine_preferences
                    .map((c) => CUISINE_LABELS[c])
                    .join(", ")}`}
                />
                {draft.allergies.length > 0 && (
                  <ReviewRow
                    label="Allergies"
                    value={draft.allergies.join(", ")}
                  />
                )}
                {draft.disliked_foods.length > 0 && (
                  <ReviewRow
                    label="Avoids"
                    value={draft.disliked_foods.join(", ")}
                  />
                )}
                <ReviewRow
                  label="Kitchen"
                  value={`${draft.meals_per_day} meals/day · ${SKILL_LABELS[draft.cooking_skill]} · ≤${draft.max_prep_minutes} min`}
                />
              </div>
            </StepShell>
          )}
        </div>

        <div className="flex gap-3 mt-8">
          {step > 0 && (
            <Button
              variant="secondary"
              size="lg"
              onClick={() => setStep((s) => s - 1)}
              disabled={submitting}
            >
              Back
            </Button>
          )}
          {step < STEPS.length - 1 ? (
            <Button
              size="lg"
              className="flex-1"
              disabled={!stepValid}
              onClick={() => setStep((s) => s + 1)}
            >
              Continue
            </Button>
          ) : (
            <Button
              size="lg"
              className="flex-1"
              loading={submitting}
              onClick={handleSubmit}
            >
              Create my plan
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------

function StepIndicator({ current }: { current: number }) {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-1.5 mb-3">
        {STEPS.map((label, i) => (
          <div
            key={label}
            className={cn(
              "h-1 flex-1 rounded-full transition-colors",
              i <= current ? "bg-brand-500" : "bg-raised",
            )}
          />
        ))}
      </div>
      <p className="text-xs font-medium text-ink-muted">
        Step {current + 1} of {STEPS.length} · {STEPS[current]}
      </p>
    </div>
  );
}

function StepShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h1 className="font-display text-2xl sm:text-3xl font-semibold text-ink">
        {title}
      </h1>
      <p className="text-ink-soft mt-2 mb-7 text-pretty">{subtitle}</p>
      {children}
    </div>
  );
}

function ReviewRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-3 border-b border-line last:border-0">
      <span className="text-sm text-ink-soft shrink-0">{label}</span>
      <span className="text-sm font-medium text-ink text-right">{value}</span>
    </div>
  );
}

/** Free-text list input rendered as removable chips. */
function TagInput({
  label,
  hint,
  values,
  onChange,
  placeholder,
}: {
  label: string;
  hint: string;
  values: string[];
  onChange: (values: string[]) => void;
  placeholder: string;
}) {
  const [input, setInput] = useState("");

  function commit() {
    const cleaned = input.trim().toLowerCase();
    if (cleaned && !values.includes(cleaned)) {
      onChange([...values, cleaned]);
    }
    setInput("");
  }

  return (
    <Field label={label} hint={hint}>
      <div className="space-y-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // Comma is a natural separator here too.
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              commit();
            }
          }}
          onBlur={commit}
          placeholder={placeholder}
        />
        {values.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {values.map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => onChange(values.filter((v) => v !== value))}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-raised text-sm text-ink hover:bg-clay-100 hover:text-clay-700 transition-colors"
              >
                {value}
                <span aria-hidden="true">×</span>
                <span className="sr-only">Remove {value}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </Field>
  );
}
