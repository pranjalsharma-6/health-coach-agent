import type { Activity } from "@/lib/types";

/** The day's training, as something you can follow rather than a category.
 *
 * "Strength training. Upper body · 45 min" tells you nothing you can act on.
 * The exercises, their sets and reps, and one form cue each are the difference
 * between a label and a session.
 */
export function SessionCard({ activity }: { activity: Activity }) {
  const isRest =
    activity.duration_minutes === 0 ||
    activity.activity_type.toLowerCase().includes("rest");

  return (
    <div className="rounded-2xl bg-raised px-4 py-3.5">
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted mb-1">
        Movement
      </p>
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
    </div>
  );
}
