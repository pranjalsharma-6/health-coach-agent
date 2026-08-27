/**
 * Display labels and descriptions for the domain enums.
 *
 * Kept out of components so the same wording appears everywhere — the
 * onboarding wizard, the profile editor and the dashboard header all read
 * from here.
 */

import type {
  ActivityLevel,
  AgentDecision,
  BudgetTier,
  CookingSkill,
  Cuisine,
  DietType,
  Goal,
  MealType,
  TrainingStyle,
} from "./types";

export const DIET_LABELS: Record<DietType, string> = {
  vegetarian: "Vegetarian",
  eggetarian: "Eggetarian",
  vegan: "Vegan",
  jain: "Jain",
  non_vegetarian: "Non-vegetarian",
  halal: "Halal",
};

export const DIET_DESCRIPTIONS: Record<DietType, string> = {
  vegetarian: "No meat, fish or egg. Dairy is fine.",
  eggetarian: "Vegetarian, but eggs are in.",
  vegan: "No animal products at all.",
  jain: "Vegetarian, and no root vegetables.",
  non_vegetarian: "Everything's on the table.",
  halal: "No pork or alcohol; halal meat only.",
};

export const DIET_EMOJI: Record<DietType, string> = {
  vegetarian: "🥗",
  eggetarian: "🍳",
  vegan: "🌱",
  jain: "🙏",
  non_vegetarian: "🍗",
  halal: "🌙",
};

export const GOAL_LABELS: Record<Goal, string> = {
  fat_loss: "Lose fat",
  muscle_gain: "Build muscle",
  maintenance: "Maintain weight",
  endurance: "Build endurance",
  general_health: "Eat healthier",
};

export const GOAL_DESCRIPTIONS: Record<Goal, string> = {
  fat_loss: "Calorie deficit with protein kept high to protect muscle.",
  muscle_gain: "Modest surplus and progressive resistance training.",
  maintenance: "Hold steady, eat well, keep it sustainable.",
  endurance: "Fuel training and recover properly.",
  general_health: "More fibre, more variety, less processed food.",
};

export const GOAL_EMOJI: Record<Goal, string> = {
  fat_loss: "📉",
  muscle_gain: "💪",
  maintenance: "⚖️",
  endurance: "🏃",
  general_health: "🌿",
};

export const CUISINE_LABELS: Record<Cuisine, string> = {
  north_indian: "North Indian",
  south_indian: "South Indian",
  continental: "Continental",
  east_asian: "East Asian",
  mediterranean: "Mediterranean",
  mixed: "A bit of everything",
};

export const TRAINING_STYLE_LABELS: Record<TrainingStyle, string> = {
  bodyweight: "Bodyweight at home",
  dumbbells: "Dumbbells at home",
  full_gym: "Full gym",
  running_cycling: "Running / cycling",
  swimming: "Swimming",
  yoga_mobility: "Yoga & mobility",
};

export const TRAINING_STYLE_DESCRIPTIONS: Record<TrainingStyle, string> = {
  bodyweight: "No equipment at all.",
  dumbbells: "A pair of dumbbells or resistance bands.",
  full_gym: "Barbells, racks and machines.",
  running_cycling: "Outdoors, on foot or on a bike.",
  swimming: "Pool access.",
  yoga_mobility: "Flexibility and recovery work.",
};

export const ACTIVITY_LABELS: Record<ActivityLevel, string> = {
  sedentary: "Sedentary",
  lightly_active: "Lightly active",
  moderately_active: "Moderately active",
  very_active: "Very active",
  extremely_active: "Extremely active",
};

export const ACTIVITY_DESCRIPTIONS: Record<ActivityLevel, string> = {
  sedentary: "Desk job, little deliberate exercise.",
  lightly_active: "Light exercise 1–3 days a week.",
  moderately_active: "Moderate exercise 3–5 days a week.",
  very_active: "Hard exercise 6–7 days a week.",
  extremely_active: "Physical job or twice-daily training.",
};

export const SKILL_LABELS: Record<CookingSkill, string> = {
  beginner: "Beginner",
  intermediate: "Comfortable",
  advanced: "Confident",
};

export const SKILL_DESCRIPTIONS: Record<CookingSkill, string> = {
  beginner: "I can boil, fry and assemble.",
  intermediate: "I cook regular meals without a recipe.",
  advanced: "I'll try anything.",
};

export const BUDGET_LABELS: Record<BudgetTier, string> = {
  low: "Tight",
  medium: "Moderate",
  high: "Flexible",
};

export const MEAL_LABELS: Record<MealType, string> = {
  breakfast: "Breakfast",
  lunch: "Lunch",
  dinner: "Dinner",
  snack: "Snack",
};

export const MEAL_EMOJI: Record<MealType, string> = {
  breakfast: "🌅",
  lunch: "☀️",
  dinner: "🌙",
  snack: "🍎",
};

export const DECISION_LABELS: Record<AgentDecision, string> = {
  no_action: "On track",
  rebalance_day: "Rebalanced your day",
  structural_replan: "Restructured your plan",
  create_initial: "Created your first plan",
};

/** Tailwind classes per decision, for timeline badges. */
export const DECISION_STYLES: Record<AgentDecision, string> = {
  no_action: "bg-sand-200 text-sand-700 dark:bg-sand-800 dark:text-sand-300",
  rebalance_day: "bg-spice-200 text-spice-600",
  structural_replan: "bg-clay-200 text-clay-700",
  create_initial: "bg-brand-200 text-brand-700",
};

export function formatRelativeDate(iso: string): string {
  const then = new Date(iso);
  const diffMs = Date.now() - then.getTime();
  const diffMinutes = Math.round(diffMs / 60000);

  if (diffMinutes < 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;

  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.round(diffHours / 24);
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;

  return then.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
