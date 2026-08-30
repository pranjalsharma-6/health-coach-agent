"use client";

import { useState } from "react";

import { Badge, Button, Spinner, cn } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { MEAL_EMOJI, MEAL_LABELS } from "@/lib/labels";
import type { MacroCheck, Meal, MealStatus, Recipe } from "@/lib/types";

interface Props {
  meal: Meal;
  status: MealStatus;
  onLog: (mealId: string, status: MealStatus) => Promise<void>;
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

  async function handleLog(next: MealStatus) {
    setLogging(next);
    try {
      await onLog(meal.meal_id, next);
    } finally {
      setLogging(null);
    }
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
            </>
          )}
          <Button size="sm" variant="ghost" onClick={toggleRecipe}>
            {recipeOpen ? "Hide recipe" : "Recipe"}
          </Button>
        </div>
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
