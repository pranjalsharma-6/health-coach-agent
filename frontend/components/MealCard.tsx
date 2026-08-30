"use client";

import { useState } from "react";

import { Badge, Button, Field, Input, Spinner, cn } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { MEAL_EMOJI, MEAL_LABELS } from "@/lib/labels";
import type {
  MacroCheck,
  Meal,
  MealStatus,
  MealSwap,
  Recipe,
} from "@/lib/types";

interface Props {
  meal: Meal;
  status: MealStatus;
  onLog: (
    mealId: string,
    status: MealStatus,
    swap?: MealSwap,
  ) => Promise<void>;
  /** Logging is only meaningful for the day the user is actually living. */
  loggable: boolean;
}

export function MealCard({ meal, status, onLog, loggable }: Props) {
  const [logging, setLogging] = useState<MealStatus | null>(null);
  const [recipe, setRecipe] = useState<Recipe | null>(meal.recipe);
  const [macroCheck, setMacroCheck] = useState<MacroCheck | null>(null);
  const [recipeOpen, setRecipeOpen] = useState(false);
  const [recipeLoading, setRecipeLoading] = useState(false);
  const [recipeError, setRecipeError] = useState<string | null>(null);
  const [swapOpen, setSwapOpen] = useState(false);
  const [swapName, setSwapName] = useState("");
  const [swapCalories, setSwapCalories] = useState("");
  const [swapProtein, setSwapProtein] = useState("");

  async function handleLog(next: MealStatus, swap?: MealSwap) {
    setLogging(next);
    try {
      await onLog(meal.meal_id, next, swap);
    } finally {
      setLogging(null);
    }
  }

  async function handleSwap() {
    const name = swapName.trim();
    if (!name) return;

    // Blank stays blank. An empty box means "I don't know", and sending 0
    // would tell the agent the meal was free, which is worse than telling it
    // nothing: it would then hand back calories the user never actually saved.
    const asNumber = (raw: string) => {
      const value = Number(raw.trim());
      return raw.trim() && Number.isFinite(value) && value >= 0
        ? Math.round(value)
        : undefined;
    };

    await handleLog("substituted", {
      name,
      calories: asNumber(swapCalories),
      protein: asNumber(swapProtein),
    });

    setSwapOpen(false);
    setSwapName("");
    setSwapCalories("");
    setSwapProtein("");
  }

  async function toggleRecipe() {
    if (recipeOpen) {
      setRecipeOpen(false);
      return;
    }
    setRecipeOpen(true);

    if (recipe) return;

    setRecipeLoading(true);
    setRecipeError(null);
    try {
      const result = await api.plans.recipe(meal.meal_id);
      setRecipe(result.recipe);
      setMacroCheck(result.macro_check);
    } catch (err) {
      setRecipeError(
        err instanceof ApiError
          ? err.message
          : "Couldn't load the recipe right now.",
      );
    } finally {
      setRecipeLoading(false);
    }
  }

  const eaten = status === "eaten" || status === "substituted";
  const skipped = status === "skipped";

  return (
    <div
      className={cn(
        "rounded-xl border transition-colors",
        eaten && "border-brand-300 bg-brand-50/50 dark:bg-brand-950/40",
        skipped && "border-clay-300 bg-clay-100/40 dark:bg-clay-900/20",
        !eaten && !skipped && "border-line bg-card",
      )}
    >
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium text-ink-muted uppercase tracking-wide">
                {MEAL_EMOJI[meal.meal_type]} {MEAL_LABELS[meal.meal_type]}
              </span>
              {eaten && (
                <Badge className="bg-brand-200 text-brand-800">
                  {status === "substituted" ? "Swapped" : "Eaten"}
                </Badge>
              )}
              {skipped && (
                <Badge className="bg-clay-200 text-clay-800">Skipped</Badge>
              )}
            </div>

            <h4
              className={cn(
                "font-medium text-ink",
                skipped && "line-through opacity-60",
              )}
            >
              {meal.name}
            </h4>
            <p className="text-sm text-ink-soft mt-1">{meal.description}</p>
          </div>

          <div className="text-right shrink-0">
            <p className="font-semibold text-ink tabular-nums">
              {meal.calories_kcal}
            </p>
            <p className="text-xs text-ink-muted">kcal</p>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-3 text-xs text-ink-muted tabular-nums">
          <span>
            <span className="font-medium text-spice-600">
              {meal.protein_g}g
            </span>{" "}
            protein
          </span>
          <span>{meal.carbs_g}g carbs</span>
          <span>{meal.fat_g}g fat</span>
        </div>

        <div className="flex flex-wrap items-center gap-2 mt-4">
          {loggable && (
            <>
              <Button
                size="sm"
                variant={eaten ? "primary" : "secondary"}
                loading={logging === "eaten"}
                onClick={() => handleLog("eaten")}
              >
                Ate it
              </Button>
              <Button
                size="sm"
                variant={skipped ? "danger" : "secondary"}
                loading={logging === "skipped"}
                onClick={() => handleLog("skipped")}
              >
                Skipped
              </Button>
              <Button
                size="sm"
                variant={status === "substituted" ? "primary" : "secondary"}
                onClick={() => setSwapOpen((open) => !open)}
                aria-expanded={swapOpen}
              >
                Ate something else
              </Button>
            </>
          )}
          <Button size="sm" variant="ghost" onClick={toggleRecipe}>
            {recipeOpen ? "Hide recipe" : "Recipe"}
          </Button>
        </div>

        {swapOpen && loggable && (
          <div className="mt-4 rounded-lg border border-line bg-raised p-4 space-y-3 animate-fade-up">
            <Field label="What did you eat?" htmlFor={`swap-name-${meal.meal_id}`}>
              <Input
                id={`swap-name-${meal.meal_id}`}
                value={swapName}
                onChange={(event) => setSwapName(event.target.value)}
                placeholder="Chole bhature at the canteen"
                maxLength={200}
              />
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Calories" htmlFor={`swap-kcal-${meal.meal_id}`}>
                <Input
                  id={`swap-kcal-${meal.meal_id}`}
                  value={swapCalories}
                  onChange={(event) => setSwapCalories(event.target.value)}
                  inputMode="numeric"
                  placeholder={String(meal.calories_kcal)}
                  className="tabular-nums"
                />
              </Field>
              <Field label="Protein (g)" htmlFor={`swap-protein-${meal.meal_id}`}>
                <Input
                  id={`swap-protein-${meal.meal_id}`}
                  value={swapProtein}
                  onChange={(event) => setSwapProtein(event.target.value)}
                  inputMode="numeric"
                  placeholder={String(meal.protein_g)}
                  className="tabular-nums"
                />
              </Field>
            </div>

            <p className="text-xs text-ink-muted">
              Leave the numbers blank if you don&apos;t know them. Kaya will
              assume the planned meal&apos;s figures, which is closer than
              pretending you ate nothing.
            </p>

            <div className="flex gap-2">
              <Button
                size="sm"
                loading={logging === "substituted"}
                disabled={!swapName.trim()}
                onClick={handleSwap}
              >
                Log it
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setSwapOpen(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        )}
      </div>

      {recipeOpen && (
        <div className="border-t border-line px-4 py-4 animate-fade-up">
          {recipeLoading && (
            <div className="flex items-center gap-2 text-sm text-ink-soft">
              <Spinner className="size-4" />
              Writing the recipe…
            </div>
          )}

          {recipeError && (
            <p className="text-sm text-clay-600">{recipeError}</p>
          )}

          {recipe && (
            <div className="space-y-4">
              <div className="flex gap-4 text-xs text-ink-muted">
                <span>⏱ {recipe.prep_minutes} min</span>
                <span>🍽 serves {recipe.serves}</span>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted mb-2">
                  Ingredients
                </p>
                <ul className="space-y-1">
                  {recipe.ingredients.map((ingredient, i) => (
                    <li key={i} className="text-sm text-ink flex gap-2">
                      <span className="text-brand-500">•</span>
                      <span>
                        {ingredient.quantity_g != null && (
                          <span className="font-medium tabular-nums">
                            {ingredient.quantity_g}g{" "}
                          </span>
                        )}
                        {ingredient.item}
                        {ingredient.preparation && (
                          <span className="text-ink-soft">
                            , {ingredient.preparation}
                          </span>
                        )}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted mb-2">
                  Method
                </p>
                <ol className="space-y-2">
                  {recipe.steps.map((step, i) => (
                    <li key={i} className="text-sm text-ink flex gap-2.5">
                      <span className="shrink-0 size-5 rounded-full bg-raised grid place-items-center text-[10px] font-semibold text-ink-soft">
                        {i + 1}
                      </span>
                      <span className="pt-0.5">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>

              {recipe.tips && (
                <p className="text-sm bg-spice-100 text-spice-600 rounded-lg px-3 py-2">
                  💡 {recipe.tips}
                </p>
              )}

              {macroCheck && <MacroCheckPanel check={macroCheck} />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


/**
 * Shows what the ingredient weights actually add up to.
 *
 * The plan asserts a meal's macros; this is the arithmetic on what's in it.
 * Surfacing both, and the gap between them, is the entire point: a number
 * the user can check beats a number they have to trust.
 */
function MacroCheckPanel({ check }: { check: MacroCheck }) {
  if (!check.reliable) {
    return (
      <div className="text-xs text-ink-muted border-t border-line pt-3">
        Kaya recognised {Math.round(check.coverage * 100)}% of these ingredients
        by weight, which isn&apos;t enough to verify the macros
        {check.unmatched.length > 0 && (
          <>: {check.unmatched.slice(0, 3).join(", ")} aren&apos;t in its table</>
        )}
        .
      </div>
    );
  }

  const kcalDrift =
    check.claimed_kcal > 0
      ? Math.abs(check.computed_kcal - check.claimed_kcal) / check.claimed_kcal
      : 0;
  const proteinDrift =
    check.claimed_protein_g > 0
      ? Math.abs(check.computed_protein_g - check.claimed_protein_g) /
        check.claimed_protein_g
      : 0;
  // Matches the backend's RECIPE_MACRO_TOLERANCE.
  const close = kcalDrift <= 0.3 && proteinDrift <= 0.3;

  return (
    <div className="border-t border-line pt-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted mb-2">
        Checked against the ingredients
      </p>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-xs text-ink-muted">Calories</p>
          <p className="text-ink tabular-nums">
            <span className="font-semibold">
              {Math.round(check.computed_kcal)}
            </span>
            <span className="text-ink-muted">
              {" "}
              vs {check.claimed_kcal} planned
            </span>
          </p>
        </div>
        <div>
          <p className="text-xs text-ink-muted">Protein</p>
          <p className="text-ink tabular-nums">
            <span className="font-semibold">
              {Math.round(check.computed_protein_g)}g
            </span>
            <span className="text-ink-muted">
              {" "}
              vs {check.claimed_protein_g}g planned
            </span>
          </p>
        </div>
      </div>
      <p
        className={cn(
          "text-xs mt-2",
          close ? "text-brand-600" : "text-spice-600",
        )}
      >
        {close
          ? "The weights add up. These portions really do hit the plan."
          : "The weights don't quite reach the plan's figures. Trust the computed numbers above over the planned ones."}
      </p>
    </div>
  );
}
