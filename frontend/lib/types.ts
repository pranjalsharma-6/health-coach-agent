/**
 * Types mirroring the backend's Pydantic models.
 *
 * Kept hand-written rather than generated: the API surface is small, and the
 * literal unions here have to match the backend enums exactly, which is easier
 * to review by eye than to trust a generator with.
 */

export type DietType =
  | "vegetarian"
  | "eggetarian"
  | "vegan"
  | "jain"
  | "non_vegetarian"
  | "halal";

export type Goal =
  | "fat_loss"
  | "muscle_gain"
  | "maintenance"
  | "endurance"
  | "general_health";

export type Cuisine =
  | "north_indian"
  | "south_indian"
  | "continental"
  | "east_asian"
  | "mediterranean"
  | "mixed";

export type ActivityLevel =
  | "sedentary"
  | "lightly_active"
  | "moderately_active"
  | "very_active"
  | "extremely_active";

export type Gender = "male" | "female" | "other";
export type CookingSkill = "beginner" | "intermediate" | "advanced";
export type BudgetTier = "low" | "medium" | "high";
export type MealType = "breakfast" | "lunch" | "dinner" | "snack";
export type MealStatus = "planned" | "eaten" | "skipped" | "substituted";

export type AgentDecision =
  | "no_action"
  | "rebalance_day"
  | "structural_replan"
  | "create_initial";

// --------------------------------------------------------------------------

export interface User {
  id: string;
  email: string;
  full_name: string;
  onboarded: boolean;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Profile {
  id?: string;
  user_id: string;
  gender: Gender;
  age_years: number;
  height_cm: number;
  current_weight_kg: number;
  target_weight_kg: number | null;
  goal: Goal;
  activity_level: ActivityLevel;
  target_timeline_weeks: number;
  diet_type: DietType;
  cuisine_preferences: Cuisine[];
  training_styles: TrainingStyle[];
  allergies: string[];
  disliked_foods: string[];
  meals_per_day: number;
  cooking_skill: CookingSkill;
  max_prep_minutes: number;
  budget_tier: BudgetTier;
  eat_out_per_week: number;
  medical_notes: string | null;
}

export type ProfileDraft = Omit<Profile, "id" | "user_id">;

export interface NutritionTargets {
  calories_kcal: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  water_ml: number;
}

export interface TargetsResponse {
  bmr_kcal: number;
  tdee_kcal: number;
  targets: NutritionTargets;
  macro_split_percent: { protein: number; carbs: number; fat: number };
  deficit_or_surplus_kcal: number;
  estimated_weekly_change_kg: number;
  safety_floor_applied: boolean;
  bmi: number;
}

export interface RecipeIngredient {
  item: string;
  /** Null for seasonings too small to weigh. Never guessed at. */
  quantity_g: number | null;
  preparation: string | null;
}

export interface Recipe {
  ingredients: RecipeIngredient[];
  steps: string[];
  prep_minutes: number;
  serves: number;
  tips: string | null;
}

/** What a recipe's weights actually add up to, versus what the plan claims. */
export interface MacroCheck {
  computed_kcal: number;
  computed_protein_g: number;
  claimed_kcal: number;
  claimed_protein_g: number;
  coverage: number;
  reliable: boolean;
  unmatched: string[];
}

export interface Meal {
  meal_id: string;
  meal_type: MealType;
  name: string;
  description: string;
  calories_kcal: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  recipe: Recipe | null;
  status: MealStatus;
  logged_at: string | null;
}

export type TrainingStyle =
  | "bodyweight"
  | "dumbbells"
  | "full_gym"
  | "running_cycling"
  | "swimming"
  | "yoga_mobility";

export interface ExercisePrescription {
  name: string;
  sets: number;
  /** Text, because reps and durations differ in kind: "8-12", "30 seconds". */
  reps: string;
  rest_seconds: number;
  cue: string | null;
}

export interface Activity {
  activity_type: string;
  duration_minutes: number;
  intensity: string;
  description: string;
  target_steps: number;
  /** Empty on a rest day, and on nothing else. */
  exercises: ExercisePrescription[];
}

export interface DailyPlan {
  day: number;
  theme: string | null;
  meals: Meal[];
  activity: Activity;
}

export interface Plan {
  id: string;
  user_id: string;
  version: number;
  is_active: boolean;
  plan_title: string;
  duration_days: number;
  agent_reasoning: string;
  daily_plans: DailyPlan[];
  targets: NutritionTargets;
  trigger: AgentDecision;
  trigger_detail: string | null;
  parent_plan_id: string | null;
  created_at: string;
}

export interface PlanSummary {
  id: string;
  version: number;
  plan_title: string;
  agent_reasoning: string;
  trigger: AgentDecision;
  trigger_detail: string | null;
  is_active: boolean;
  created_at: string;
}

export interface AdherenceSnapshot {
  date: string;
  /** 1-based day of the plan this date maps to; null with no active plan. */
  plan_day: number | null;
  meals_planned: number;
  meals_eaten: number;
  meals_skipped: number;
  meals_pending: number;
  calories_target: number;
  calories_consumed: number;
  calories_remaining: number;
  protein_target_g: number;
  protein_consumed_g: number;
  protein_remaining_g: number;
  steps: number | null;
  sleep_hours: number | null;
  skip_streak_days: number;
  skips_last_7_days: number;
  adherence_rate_7d: number;
  meals_logged_7d: number;
}

export interface MealLogEntry {
  meal_id: string;
  status: MealStatus;
  actual_calories_kcal: number | null;
  actual_protein_g: number | null;
  substitute_name: string | null;
  note: string | null;
  logged_at: string;
}

export interface DailyLog {
  id?: string;
  user_id: string;
  log_date: string;
  meals: MealLogEntry[];
  weight_kg: number | null;
  steps: number | null;
  sleep_hours: number | null;
  water_ml: number | null;
}

export interface MealLogResponse {
  log: DailyLog;
  snapshot: AdherenceSnapshot;
  agent_recommended: boolean;
  agent_reason: string | null;
}

export interface AgentEvent {
  id?: string;
  user_id: string;
  decision: AgentDecision;
  rationale: string;
  trigger_summary: string;
  snapshot: AdherenceSnapshot | null;
  resulting_plan_id: string | null;
  created_at: string;
}

/** One streamed step from the agent's run. */
export interface AgentStep {
  node: string;
  status: "running" | "done" | "failed" | "error";
  message: string;
  attempt?: number;
  decision?: string;
  /** Validation failures, on a `validate` step that rejected a plan. */
  errors?: string[];
  /** Run-level failure, present only on the terminal `complete` step. */
  error?: string | null;
  plan_id?: string | null;
  duration_ms?: number;
  version?: number;
}

export interface WeightPoint {
  date: string;
  weight_kg: number;
}
