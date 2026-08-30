"use client";

import { useState, type FormEvent } from "react";

import { Alert, Button, Field, Input } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import type { DailyLog } from "@/lib/types";

interface Props {
  today: DailyLog | null;
  onSaved: (log: DailyLog) => void;
}

/**
 * Records the day's body and activity metrics.
 *
 * These aren't decoration: `steps` and `sleep_hours` are read straight into the
 * agent's evidence block, and `weight_kg` is the only source the progress chart
 * draws from. Without this form the sensing half of the loop has no input.
 */
export function MetricsLogger({ today, onSaved }: Props) {
  const [weight, setWeight] = useState(today?.weight_kg?.toString() ?? "");
  const [steps, setSteps] = useState(today?.steps?.toString() ?? "");
  const [sleep, setSleep] = useState(today?.sleep_hours?.toString() ?? "");
  const [water, setWater] = useState(today?.water_ml?.toString() ?? "");

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSaved(false);

    // Only send fields the user actually filled in. An empty box means
    // "no reading", not zero.
    const payload: Record<string, number> = {};
    if (weight.trim()) payload.weight_kg = Number(weight);
    if (steps.trim()) payload.steps = Number(steps);
    if (sleep.trim()) payload.sleep_hours = Number(sleep);
    if (water.trim()) payload.water_ml = Number(water);

    if (Object.keys(payload).length === 0) {
      setError("Fill in at least one value.");
      return;
    }
    if (Object.values(payload).some((v) => !Number.isFinite(v))) {
      setError("Those need to be numbers.");
      return;
    }

    setSaving(true);
    try {
      const log = await api.logs.logMetrics(payload);
      onSaved(log);
      setSaved(true);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Couldn't save those.",
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="px-5 pb-5 space-y-4">
      {error && <Alert tone="error">{error}</Alert>}
      {saved && !error && <Alert tone="success">Saved.</Alert>}

      <div className="grid grid-cols-2 gap-3">
        <Field label="Weight" hint="kg" htmlFor="m-weight">
          <Input
            id="m-weight"
            type="number"
            step="0.1"
            min={30}
            max={300}
            inputMode="decimal"
            value={weight}
            onChange={(e) => setWeight(e.target.value)}
            placeholder="72.5"
          />
        </Field>

        <Field label="Steps" hint="today" htmlFor="m-steps">
          <Input
            id="m-steps"
            type="number"
            min={0}
            max={100000}
            inputMode="numeric"
            value={steps}
            onChange={(e) => setSteps(e.target.value)}
            placeholder="8000"
          />
        </Field>

        <Field label="Sleep" hint="hours" htmlFor="m-sleep">
          <Input
            id="m-sleep"
            type="number"
            step="0.5"
            min={0}
            max={24}
            inputMode="decimal"
            value={sleep}
            onChange={(e) => setSleep(e.target.value)}
            placeholder="7.5"
          />
        </Field>

        <Field label="Water" hint="ml" htmlFor="m-water">
          <Input
            id="m-water"
            type="number"
            step="100"
            min={0}
            max={10000}
            inputMode="numeric"
            value={water}
            onChange={(e) => setWater(e.target.value)}
            placeholder="2500"
          />
        </Field>
      </div>

      <Button type="submit" loading={saving} className="w-full">
        Save today&apos;s numbers
      </Button>

      <p className="text-xs text-ink-muted">
        Kaya reads your steps and sleep when it decides whether to change your
        plan. Leave anything blank that you don&apos;t track.
      </p>
    </form>
  );
}
