"use client";

import { useState } from "react";

import { Badge, Button, cn } from "@/components/ui";
import type { Activity, SessionStatus } from "@/lib/types";

/** The day's training, as something you can follow rather than a category.
 *
 * "Strength training. Upper body · 45 min" tells you nothing you can act on.
 * The exercises, their sets and reps, and one form cue each are the difference
 * between a label and a session.
 */
export function SessionCard({
  activity,
  planDay,
  status = "planned",
  onLog,
  loggable = false,
}: {
  activity: Activity;
  planDay: number;
  status?: SessionStatus;
  onLog?: (planDay: number, status: SessionStatus) => Promise<void>;
  loggable?: boolean;
}) {
  const [logging, setLogging] = useState<SessionStatus | null>(null);

  const isRest =
    activity.duration_minutes === 0 ||
    activity.activity_type.toLowerCase().includes("rest");

  async function handleLog(next: SessionStatus) {
    if (!onLog) return;
    setLogging(next);
    try {
      await onLog(planDay, next);
    } finally {
      setLogging(null);
    }
  }

  return (
    <div
      className={cn(
        "rounded-2xl px-4 py-3.5",
        status === "done" && "bg-brand-50/60 dark:bg-brand-950/40",
        status === "skipped" && "bg-clay-100/50 dark:bg-clay-900/20",
        status === "planned" && "bg-raised",
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
          Movement
        </p>
        {status === "done" && (
          <Badge className="bg-brand-200 text-brand-800">Done</Badge>
        )}
        {status === "skipped" && (
          <Badge className="bg-clay-200 text-clay-800">Skipped</Badge>
        )}
      </div>
      <p className="font-medium text-ink">
        {activity.activity_type}
        {activity.duration_minutes > 0 && (
          <span className="text-ink-soft font-normal">
            {" "}
            · {activity.duration_minutes} min
          </span>
        )}
      </p>
      <p className="text-sm text-ink-soft mt-0.5">{activity.description}</p>

      {!isRest && activity.exercises.length > 0 && (
        <ol className="mt-3 space-y-2.5 border-t border-line pt-3">
          {activity.exercises.map((exercise, i) => (
            <li key={`${exercise.name}-${i}`} className="min-w-0">
              <div className="flex items-baseline justify-between gap-3 flex-wrap">
                <span className="font-medium text-ink text-sm">
                  {exercise.name}
                </span>
                <span className="text-xs text-ink-soft whitespace-nowrap">
                  {exercise.sets} × {exercise.reps}
                  {exercise.rest_seconds > 0 && (
                    <span className="text-ink-muted">
                      {" · "}
                      {exercise.rest_seconds}s rest
                    </span>
                  )}
                </span>
              </div>
              {exercise.cue && (
                <p className="text-xs text-ink-muted mt-0.5 text-pretty">
                  {exercise.cue}
                </p>
              )}
            </li>
          ))}
        </ol>
      )}

      {/* A rest day has nothing to skip, so it gets no buttons. Offering them
          would invite a skip streak made of days that asked for nothing. */}
      {loggable && !isRest && (
        <div className="flex flex-wrap items-center gap-2 mt-3 border-t border-line pt-3">
          <Button
            size="sm"
            variant={status === "done" ? "primary" : "secondary"}
            loading={logging === "done"}
            onClick={() => handleLog("done")}
          >
            Did it
          </Button>
          <Button
            size="sm"
            variant={status === "skipped" ? "danger" : "secondary"}
            loading={logging === "skipped"}
            onClick={() => handleLog("skipped")}
          >
            Skipped
          </Button>
        </div>
      )}
    </div>
  );
}
