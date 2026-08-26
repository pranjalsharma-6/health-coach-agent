"use client";

import { useCallback, useRef, useState } from "react";

import { Alert, Button, cn } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { DECISION_LABELS } from "@/lib/labels";
import type { AgentDecision, AgentStep } from "@/lib/types";

const NODE_LABELS: Record<string, string> = {
  sense: "Reading your logs",
  evaluate: "Measuring your day",
  decide: "Deciding what to do",
  generate: "Writing the plan",
  validate: "Checking it's safe",
  persist: "Saving",
  record: "Wrapping up",
  complete: "Done",
  error: "Failed",
};

const NODE_ICONS: Record<string, string> = {
  sense: "👀",
  evaluate: "📊",
  decide: "🧭",
  generate: "✍️",
  validate: "🛡️",
  persist: "💾",
  record: "📝",
  complete: "✅",
  error: "⚠️",
};

interface Props {
  onComplete: (result: { decision?: string; planId?: string | null }) => void;
  /** Rendered above the button — e.g. "you skipped lunch, want a rebalance?" */
  prompt?: string | null;
}

export function AgentRunner({ onComplete, prompt }: Props) {
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [finished, setFinished] = useState<AgentStep | null>(null);

  // The prompt that was showing when the current run started. Comparing against
  // it tells us whether an incoming recommendation is newer than the last run's
  // result — without which `finished` sticks around and permanently suppresses
  // every later prompt, so logging a skip after a successful run looks like
  // nothing happened. Derived rather than synced in an effect.
  const [promptAtRun, setPromptAtRun] = useState<string | null>(null);

  // Guards against a double-click starting two concurrent runs.
  const runningRef = useRef(false);

  const run = useCallback(
    async (forceReplan: boolean) => {
      if (runningRef.current) return;
      runningRef.current = true;

      setRunning(true);
      setError(null);
      setSteps([]);
      setFinished(null);
      setPromptAtRun(prompt ?? null);

      try {
        for await (const step of api.agent.stream(forceReplan)) {
          if (step.node === "complete") {
            setFinished(step);
            onComplete({ decision: step.decision, planId: step.plan_id });
            if (step.error) setError(step.error);
          } else {
            // Replace the "running" entry for a node with its final state,
            // so the list reads as one line per step rather than two.
            setSteps((prev) => {
              const withoutPending = prev.filter(
                (s) => !(s.node === step.node && s.status === "running"),
              );
              return [...withoutPending, step];
            });
          }
        }
      } catch (err) {
        setError(
          err instanceof ApiError
            ? err.message
            : "The agent run failed. Please try again.",
        );
      } finally {
        setRunning(false);
        runningRef.current = false;
      }
    },
    [onComplete, prompt],
  );

  const decision = finished?.decision as AgentDecision | undefined;

  // A recommendation the current result doesn't already account for.
  const promptIsNew = prompt != null && prompt !== promptAtRun;

  return (
    <div className="space-y-4">
      {promptIsNew && !running && (
        <Alert tone="warning" title="Kaya noticed something">
          {prompt}
        </Alert>
      )}

      <div className="flex flex-wrap gap-2">
        <Button onClick={() => run(false)} loading={running} size="lg">
          {running ? "Kaya is working…" : "Run Kaya"}
        </Button>
        <Button
          onClick={() => run(true)}
          disabled={running}
          variant="secondary"
          size="lg"
        >
          Rebuild from scratch
        </Button>
      </div>

      {error && <Alert tone="error">{error}</Alert>}

      {steps.length > 0 && (
        <ol className="space-y-1 border-l-2 border-line ml-3 pl-5 py-1">
          {steps.map((step, i) => (
            <li
              key={`${step.node}-${i}`}
              className="relative animate-fade-up py-1.5"
            >
              <span
                className={cn(
                  "absolute -left-[27px] top-2 size-4 rounded-full grid place-items-center text-[9px]",
                  step.status === "failed" || step.status === "error"
                    ? "bg-clay-500"
                    : step.status === "running"
                      ? "bg-spice-400 animate-pulse-ring"
                      : "bg-brand-500",
                )}
                aria-hidden="true"
              />
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className="text-xs font-semibold text-ink">
                  {NODE_ICONS[step.node] ?? "•"}{" "}
                  {NODE_LABELS[step.node] ?? step.node}
                </span>
                {step.attempt && step.attempt > 1 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-spice-100 text-spice-600 font-medium">
                    attempt {step.attempt}
                  </span>
                )}
              </div>
              <p
                className={cn(
                  "text-sm mt-0.5",
                  step.status === "failed" || step.status === "error"
                    ? "text-clay-600"
                    : "text-ink-soft",
                )}
              >
                {step.message}
              </p>
            </li>
          ))}
        </ol>
      )}

      {finished && !promptIsNew && !error && decision && (
        <Alert tone="success" title={DECISION_LABELS[decision] ?? "Done"}>
          {decision === "no_action"
            ? "Nothing needed changing — your plan still fits."
            : "Your plan has been updated. Scroll down to see it."}
        </Alert>
      )}
    </div>
  );
}
