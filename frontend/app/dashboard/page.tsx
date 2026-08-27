"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { AgentRunner } from "@/components/AgentRunner";
import { MealCard } from "@/components/MealCard";
import { MetricsLogger } from "@/components/MetricsLogger";
import { WeightChart } from "@/components/WeightChart";
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  ProgressBar,
  Skeleton,
  cn,
} from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { useAuth, useRequireAuth } from "@/lib/auth";
import {
  DECISION_LABELS,
  DECISION_STYLES,
  DIET_EMOJI,
  DIET_LABELS,
  GOAL_LABELS,
  formatRelativeDate,
} from "@/lib/labels";
import type {
  AdherenceSnapshot,
  AgentEvent,
  DailyLog,
  MealStatus,
  Plan,
  Profile,
  TargetsResponse,
  WeightPoint,
} from "@/lib/types";

interface DashboardData {
  profile: Profile;
  targets: TargetsResponse;
  plan: Plan | null;
  log: DailyLog;
  snapshot: AdherenceSnapshot;
  events: AgentEvent[];
  weights: WeightPoint[];
}

export default function DashboardPage() {
  const { loading: authLoading, user } = useRequireAuth();
  const { logout } = useAuth();

  const [profile, setProfile] = useState<Profile | null>(null);
  const [targets, setTargets] = useState<TargetsResponse | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [log, setLog] = useState<DailyLog | null>(null);
  const [snapshot, setSnapshot] = useState<AdherenceSnapshot | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [weights, setWeights] = useState<WeightPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [agentPrompt, setAgentPrompt] = useState<string | null>(null);
  const [dayOverride, setDayOverride] = useState<number | null>(null);

  /** Fetch everything the dashboard needs. Pure IO — touches no state. */
  const fetchDashboard = useCallback(async (): Promise<DashboardData> => {
    // One round trip's worth of latency instead of six.
    const [profile, targets, plan, log, snapshot, events, weights] =
      await Promise.all([
        api.profile.get(),
        api.profile.targets(),
        api.plans.active(),
        api.logs.today(),
        api.logs.adherence(),
        api.agent.events(),
        api.logs.weight(),
      ]);
    return { profile, targets, plan, log, snapshot, events, weights };
  }, []);

  const applyDashboard = useCallback((data: DashboardData) => {
    setProfile(data.profile);
    setTargets(data.targets);
    setPlan(data.plan);
    setLog(data.log);
    setSnapshot(data.snapshot);
    setEvents(data.events);
    setWeights(data.weights);
    setError(null);
  }, []);

  useEffect(() => {
    if (authLoading || !user) return;

    // Guards against applying a response that lands after unmount, or after a
    // newer load has already superseded this one.
    let cancelled = false;

    fetchDashboard()
      .then((data) => {
        if (!cancelled) applyDashboard(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? err.message
            : "Couldn't load your dashboard.",
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [authLoading, user, fetchDashboard, applyDashboard]);

  /** Which day of the plan corresponds to today. */
  const todayIndex = useMemo(() => {
    if (!plan?.daily_plans.length) return 0;
    const created = new Date(plan.created_at);
    const elapsed = Math.floor(
      (Date.now() - created.getTime()) / (1000 * 60 * 60 * 24),
    );
    return Math.max(0, elapsed) % plan.daily_plans.length;
  }, [plan]);

  // The tab defaults to today and only diverges once the user picks another
  // day. Deriving it from an optional override avoids syncing state in an
  // effect, and the clamp keeps a stale pick valid if a shorter plan arrives.
  const selectedDay = useMemo(() => {
    if (dayOverride === null) return todayIndex;
    const lastIndex = Math.max((plan?.daily_plans.length ?? 1) - 1, 0);
    return Math.min(dayOverride, lastIndex);
  }, [dayOverride, todayIndex, plan]);

  const statusByMeal = useMemo(() => {
    const map = new Map<string, MealStatus>();
    log?.meals.forEach((entry) => map.set(entry.meal_id, entry.status));
    return map;
  }, [log]);

  const handleLogMeal = useCallback(
    async (mealId: string, status: MealStatus) => {
      try {
        const result = await api.logs.logMeal({ meal_id: mealId, status });
        setLog(result.log);
        setSnapshot(result.snapshot);
        setAgentPrompt(result.agent_recommended ? result.agent_reason : null);
      } catch (err) {
        setError(
          err instanceof ApiError ? err.message : "Couldn't save that.",
        );
      }
    },
    [],
  );

  const handleAgentComplete = useCallback(async () => {
    setAgentPrompt(null);
    // A new plan should land the user back on today rather than whichever tab
    // they were browsing. Reset in the handler, not an effect.
    setDayOverride(null);
    try {
      applyDashboard(await fetchDashboard());
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Couldn't refresh your plan.",
      );
    }
  }, [fetchDashboard, applyDashboard]);

  if (authLoading || loading) return <DashboardSkeleton />;

  const day = plan?.daily_plans[selectedDay];
  const isToday = selectedDay === todayIndex;

  return (
    <div className="min-h-screen">
      <header className="border-b border-line sticky top-0 bg-page/90 backdrop-blur z-10">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 min-w-0">
            <div className="size-8 rounded-lg bg-brand-600 grid place-items-center text-white font-display font-bold shrink-0">
              K
            </div>
            <div className="min-w-0">
              <p className="font-medium text-sm text-ink truncate">
                {user?.full_name}
              </p>
              {profile && (
                <p className="text-xs text-ink-muted truncate">
                  {DIET_EMOJI[profile.diet_type]}{" "}
                  {DIET_LABELS[profile.diet_type]} ·{" "}
                  {GOAL_LABELS[profile.goal]}
                </p>
              )}
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={logout}>
            Sign out
          </Button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        {error && <Alert tone="error">{error}</Alert>}

        {targets?.safety_floor_applied && (
          <Alert tone="warning" title="Your target was raised for safety">
            The timeline you set implied a calorie target below what&apos;s safe,
            so Kaya raised it. You&apos;ll lose weight slightly slower than
            planned — deliberately.
          </Alert>
        )}

        {/* `min-w-0` on the grid children matters: grid items default to
            min-width:auto, so the day-tab row's intrinsic width would force the
            track wider than the viewport and scroll the whole page sideways on
            mobile, instead of letting the tab strip scroll on its own. */}
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 min-w-0 space-y-6">
            {snapshot && targets && (
              <Card>
                <CardHeader
                  title="Today"
                  subtitle={
                    snapshot.meals_planned > 0
                      ? [
                          `${snapshot.meals_eaten} of ${snapshot.meals_planned} eaten`,
                          // A skip is logged too — don't imply nothing happened.
                          snapshot.meals_skipped > 0 &&
                            `${snapshot.meals_skipped} skipped`,
                          snapshot.meals_pending > 0 &&
                            `${snapshot.meals_pending} to go`,
                        ]
                          .filter(Boolean)
                          .join(" · ")
                      : "No plan for today yet"
                  }
                  action={
                    snapshot.skip_streak_days > 0 ? (
                      <Badge className="bg-clay-200 text-clay-800">
                        {snapshot.skip_streak_days}-day skip streak
                      </Badge>
                    ) : snapshot.meals_logged_7d > 0 ? (
                      <Badge className="bg-brand-200 text-brand-800">
                        {Math.round(snapshot.adherence_rate_7d * 100)}% this week
                      </Badge>
                    ) : null
                  }
                />
                <div className="px-5 pb-5 space-y-4">
                  <ProgressBar
                    label="Calories"
                    unit="kcal"
                    value={snapshot.calories_consumed}
                    max={snapshot.calories_target}
                  />
                  <ProgressBar
                    label="Protein"
                    unit="g"
                    tone="spice"
                    value={snapshot.protein_consumed_g}
                    max={snapshot.protein_target_g}
                  />

                  <div className="grid grid-cols-3 gap-3 pt-2">
                    <Stat
                      label="BMR"
                      value={targets.bmr_kcal}
                      unit="kcal"
                    />
                    <Stat
                      label="Burn"
                      value={targets.tdee_kcal}
                      unit="kcal/day"
                    />
                    <Stat
                      label="Projected"
                      value={targets.estimated_weekly_change_kg}
                      unit="kg/week"
                      signed
                    />
                  </div>
                </div>
              </Card>
            )}

            {plan && day ? (
              <Card>
                <CardHeader
                  title={plan.plan_title}
                  subtitle={plan.agent_reasoning}
                  action={<Badge>v{plan.version}</Badge>}
                />

                <div className="px-5 pb-2 flex gap-1.5 overflow-x-auto">
                  {plan.daily_plans.map((d, i) => (
                    <button
                      key={d.day}
                      onClick={() => setDayOverride(i)}
                      className={cn(
                        "px-3 py-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-colors",
                        i === selectedDay
                          ? "bg-brand-600 text-white"
                          : "bg-raised text-ink-soft hover:text-ink",
                      )}
                    >
                      Day {d.day}
                      {i === todayIndex && (
                        <span className="ml-1.5 text-[10px] opacity-80">
                          today
                        </span>
                      )}
                    </button>
                  ))}
                </div>

                <div className="px-5 py-4 space-y-3">
                  <div className="rounded-xl bg-raised px-4 py-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted mb-1">
                      Movement
                    </p>
                    <p className="font-medium text-ink">
                      {day.activity.activity_type}
                      {day.activity.duration_minutes > 0 && (
                        <span className="text-ink-soft font-normal">
                          {" "}
                          · {day.activity.duration_minutes} min
                        </span>
                      )}
                    </p>
                    <p className="text-sm text-ink-soft mt-0.5">
                      {day.activity.description}
                    </p>
                  </div>

                  {day.meals.map((meal) => (
                    <MealCard
                      key={meal.meal_id}
                      meal={meal}
                      status={statusByMeal.get(meal.meal_id) ?? "planned"}
                      onLog={handleLogMeal}
                      loggable={isToday}
                    />
                  ))}

                  {!isToday && (
                    <p className="text-xs text-ink-muted text-center pt-1">
                      You can only log meals for today.
                    </p>
                  )}
                </div>
              </Card>
            ) : (
              <Card>
                <EmptyState
                  emoji="🍳"
                  title="No plan yet"
                  description="Run Kaya and it'll build your first week from the profile you just filled in."
                />
              </Card>
            )}
          </div>

          <div className="min-w-0 space-y-6">
            <Card>
              <CardHeader
                title="Kaya"
                subtitle="Runs the sense → evaluate → decide → act loop."
              />
              <div className="px-5 pb-5">
                <AgentRunner
                  onComplete={handleAgentComplete}
                  prompt={agentPrompt}
                />
              </div>
            </Card>

            <Card>
              <CardHeader
                title="Progress"
                subtitle="Your real weigh-ins — nothing simulated."
              />
              <WeightChart
                points={weights}
                targetKg={profile?.target_weight_kg ?? null}
              />
            </Card>

            <Card>
              <CardHeader
                title="Today's numbers"
                subtitle="Steps and sleep feed straight into Kaya's decision."
              />
              <MetricsLogger
                today={log}
                onSaved={(updated) => {
                  setLog(updated);
                  // Reflect a new weigh-in in the chart without a full reload.
                  if (updated.weight_kg != null) {
                    setWeights((prev) => {
                      const others = prev.filter(
                        (p) => p.date !== updated.log_date,
                      );
                      return [
                        ...others,
                        {
                          date: updated.log_date,
                          weight_kg: updated.weight_kg as number,
                        },
                      ].sort((a, b) => a.date.localeCompare(b.date));
                    });
                  }
                }}
              />
            </Card>

            <Card>
              <CardHeader
                title="Decision history"
                subtitle="Every run, including the ones that changed nothing."
              />
              <div className="px-5 pb-5">
                {events.length === 0 ? (
                  <p className="text-sm text-ink-muted py-4 text-center">
                    Nothing yet. Run Kaya to get started.
                  </p>
                ) : (
                  <ol className="space-y-4">
                    {events.slice(0, 12).map((event, i) => (
                      <li key={event.id ?? i} className="flex gap-3">
                        <div className="flex flex-col items-center shrink-0">
                          <span className="size-2 rounded-full bg-brand-400 mt-1.5" />
                          {i < Math.min(events.length, 12) - 1 && (
                            <span className="w-px flex-1 bg-line mt-1" />
                          )}
                        </div>
                        <div className="min-w-0 pb-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <Badge className={DECISION_STYLES[event.decision]}>
                              {DECISION_LABELS[event.decision]}
                            </Badge>
                            <span className="text-xs text-ink-muted">
                              {formatRelativeDate(event.created_at)}
                            </span>
                          </div>
                          <p className="text-sm text-ink-soft mt-1.5">
                            {event.trigger_summary || event.rationale}
                          </p>
                        </div>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

function Stat({
  label,
  value,
  unit,
  signed = false,
}: {
  label: string;
  value: number;
  unit: string;
  signed?: boolean;
}) {
  const display = signed && value > 0 ? `+${value}` : String(value);
  return (
    <div className="rounded-xl bg-raised px-3 py-2.5">
      <p className="text-xs text-ink-muted">{label}</p>
      <p className="font-semibold text-ink tabular-nums mt-0.5">{display}</p>
      <p className="text-[10px] text-ink-muted">{unit}</p>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="min-h-screen">
      <div className="border-b border-line h-16" />
      <div className="max-w-6xl mx-auto px-6 py-8 grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Skeleton className="h-48" />
          <Skeleton className="h-96" />
        </div>
        <div className="space-y-6">
          <Skeleton className="h-56" />
          <Skeleton className="h-72" />
        </div>
      </div>
    </div>
  );
}
