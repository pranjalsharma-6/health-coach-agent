"use client";

/**
 * Weight trend — a single-series line chart over time.
 *
 * Deliberately hand-rolled SVG rather than a charting library: one series and a
 * reference line don't justify 40kB of dependency, and it keeps the mark specs
 * (2px stroke, 8px end marker with a surface ring, hairline recessive grid)
 * under direct control.
 *
 * Series colour is `#5d9174` (Kaya brand-500) in both themes — it clears the
 * lightness band, chroma floor and 3:1 contrast against the light card (#ffffff)
 * and the dark card (#1c1a16). The deeper brand-600 reads gray (chroma 0.089)
 * and was rejected.
 *
 * The target weight is drawn as a recessive dashed reference line, not a second
 * series: it's a goal, not data, and giving it a hue would spend the identity
 * channel on something the reader already understands from position.
 */

import { useMemo, useState } from "react";

import { EmptyState, cn } from "@/components/ui";
import type { WeightPoint } from "@/lib/types";

const SERIES = "#5d9174";

// viewBox units, sized to the sidebar column it lives in (~380px) so the 11px
// axis text renders at roughly its nominal size instead of being scaled down to
// an illegible 6px. Strokes use non-scaling-stroke to stay true at any width.
const W = 380;
const H = 210;
const PAD = { top: 14, right: 46, bottom: 26, left: 36 };

const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

interface Props {
  points: WeightPoint[];
  targetKg?: number | null;
}

export function WeightChart({ points, targetKg }: Props) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);

  const data = useMemo(
    () =>
      [...points]
        .map((p) => ({ ...p, t: new Date(p.date).getTime() }))
        .filter((p) => Number.isFinite(p.t))
        .sort((a, b) => a.t - b.t),
    [points],
  );

  const scale = useMemo(() => {
    if (data.length === 0) return null;

    const weights = data.map((d) => d.weight_kg);
    const candidates = targetKg != null ? [...weights, targetKg] : weights;
    let lo = Math.min(...candidates);
    let hi = Math.max(...candidates);

    // A flat series would collapse to a zero-height range.
    if (hi - lo < 1) {
      lo -= 1;
      hi += 1;
    }
    const pad = (hi - lo) * 0.15;
    lo -= pad;
    hi += pad;

    const t0 = data[0].t;
    const t1 = data[data.length - 1].t;
    const span = t1 - t0 || 1;

    return {
      lo,
      hi,
      x: (t: number) =>
        data.length === 1
          ? PAD.left + PLOT_W / 2
          : PAD.left + ((t - t0) / span) * PLOT_W,
      y: (kg: number) => PAD.top + PLOT_H - ((kg - lo) / (hi - lo)) * PLOT_H,
    };
  }, [data, targetKg]);

  // One point is a current value, not a trend — a one-point line chart is a
  // form error. Fall back to a stat tile.
  if (data.length === 0) {
    return (
      <EmptyState
        emoji="⚖️"
        title="No weigh-ins yet"
        description="Log your weight below. Once there are a few entries, your trend appears here — real data only, never a simulated line."
      />
    );
  }

  if (data.length === 1 || !scale) {
    const only = data[0];
    return (
      <div className="px-5 pb-5">
        <p className="text-xs text-ink-muted">Latest weigh-in</p>
        <p className="text-4xl font-semibold text-ink mt-1">
          {only.weight_kg}
          <span className="text-lg text-ink-muted ml-1">kg</span>
        </p>
        <p className="text-sm text-ink-soft mt-2">
          {formatDate(only.date)} · log another to start the trend line.
        </p>
      </div>
    );
  }

  const first = data[0];
  const last = data[data.length - 1];
  const change = +(last.weight_kg - first.weight_kg).toFixed(1);

  const ticks = niceTicks(scale.lo, scale.hi, 4);
  const path = data
    .map((d, i) => `${i === 0 ? "M" : "L"}${scale.x(d.t)},${scale.y(d.weight_kg)}`)
    .join(" ");

  const active = hoverIndex != null ? data[hoverIndex] : null;

  return (
    <div className="px-5 pb-5">
      <div className="flex items-baseline gap-3 mb-3">
        <p className="text-3xl font-semibold text-ink">
          {last.weight_kg}
          <span className="text-base text-ink-muted ml-1">kg</span>
        </p>
        <p
          className={cn(
            "text-sm font-medium tabular-nums",
            change < 0 ? "text-brand-600" : change > 0 ? "text-clay-600" : "text-ink-muted",
          )}
        >
          {change > 0 ? "+" : ""}
          {change} kg
          <span className="text-ink-muted font-normal"> since {formatDate(first.date)}</span>
        </p>
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Weight trend from ${last.weight_kg} kilograms, ${data.length} entries`}
          // Sizing via CSS, not attributes: height="auto" is not a valid SVG
          // length, and the browser derives the ratio from the viewBox anyway.
          className="w-full h-auto overflow-visible"
          onPointerLeave={() => setHoverIndex(null)}
          onPointerMove={(event) => {
            const svg = event.currentTarget;
            const rect = svg.getBoundingClientRect();
            // Map client px back into viewBox units.
            const vx = ((event.clientX - rect.left) / rect.width) * W;
            let nearest = 0;
            let best = Infinity;
            data.forEach((d, i) => {
              const dist = Math.abs(scale.x(d.t) - vx);
              if (dist < best) {
                best = dist;
                nearest = i;
              }
            });
            setHoverIndex(nearest);
          }}
        >
          {/* Gridlines — hairline, solid, recessive. */}
          {ticks.map((tick) => (
            <g key={tick}>
              <line
                x1={PAD.left}
                x2={PAD.left + PLOT_W}
                y1={scale.y(tick)}
                y2={scale.y(tick)}
                stroke="var(--surface-border)"
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
              />
              <text
                x={PAD.left - 8}
                y={scale.y(tick)}
                textAnchor="end"
                dominantBaseline="middle"
                className="fill-[var(--text-muted)] text-[11px] tabular-nums"
              >
                {tick}
              </text>
            </g>
          ))}

          {/* Target — a goal, so recessive and dashed, never a second series. */}
          {targetKg != null && targetKg >= scale.lo && targetKg <= scale.hi && (
            <g>
              <line
                x1={PAD.left}
                x2={PAD.left + PLOT_W}
                y1={scale.y(targetKg)}
                y2={scale.y(targetKg)}
                stroke="var(--text-muted)"
                strokeWidth={1}
                strokeDasharray="5 4"
                vectorEffect="non-scaling-stroke"
              />
              <text
                x={PAD.left + PLOT_W + 6}
                y={scale.y(targetKg)}
                dominantBaseline="middle"
                className="fill-[var(--text-muted)] text-[11px]"
              >
                target
              </text>
            </g>
          )}

          {/* Crosshair — readers aim at a date, not at a 2px line. */}
          {active && (
            <line
              x1={scale.x(active.t)}
              x2={scale.x(active.t)}
              y1={PAD.top}
              y2={PAD.top + PLOT_H}
              stroke="var(--text-muted)"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
          )}

          <path
            d={path}
            fill="none"
            stroke={SERIES}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />

          {/* End marker: r>=4 with a 2px surface ring so it reads over the line. */}
          <circle
            cx={scale.x(last.t)}
            cy={scale.y(last.weight_kg)}
            r={4.5}
            fill={SERIES}
            stroke="var(--surface-card)"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />

          {active && active !== last && (
            <circle
              cx={scale.x(active.t)}
              cy={scale.y(active.weight_kg)}
              r={4.5}
              fill={SERIES}
              stroke="var(--surface-card)"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          )}

          {/* Value at the line end — labelled selectively, never every point. */}
          <text
            x={scale.x(last.t) + 10}
            y={scale.y(last.weight_kg)}
            dominantBaseline="middle"
            className="fill-[var(--text-primary)] text-[12px] font-semibold tabular-nums"
          >
            {last.weight_kg}
          </text>

          <text
            x={PAD.left}
            y={H - 8}
            className="fill-[var(--text-muted)] text-[11px]"
          >
            {formatDate(first.date)}
          </text>
          <text
            x={PAD.left + PLOT_W}
            y={H - 8}
            textAnchor="end"
            className="fill-[var(--text-muted)] text-[11px]"
          >
            {formatDate(last.date)}
          </text>
        </svg>

        {active && (
          <div
            className="pointer-events-none absolute -translate-x-1/2 -translate-y-full bg-card border border-line rounded-lg px-2.5 py-1.5 shadow-md whitespace-nowrap"
            style={{
              left: `${(scale.x(active.t) / W) * 100}%`,
              top: `${(scale.y(active.weight_kg) / H) * 100}%`,
            }}
          >
            {/* Value leads, label follows — the reader has the series already. */}
            <p className="text-sm font-semibold text-ink tabular-nums">
              {active.weight_kg} kg
            </p>
            <p className="text-[11px] text-ink-muted">{formatDate(active.date)}</p>
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={() => setShowTable((v) => !v)}
        className="text-xs text-ink-muted hover:text-ink mt-3 underline underline-offset-2"
      >
        {showTable ? "Hide" : "Show"} data table
      </button>

      {showTable && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">Recorded weigh-ins</caption>
            <thead>
              <tr className="text-left text-ink-muted border-b border-line">
                <th scope="col" className="py-1.5 font-medium">Date</th>
                <th scope="col" className="py-1.5 font-medium text-right">Weight</th>
              </tr>
            </thead>
            <tbody>
              {[...data].reverse().map((d) => (
                <tr key={d.date} className="border-b border-line last:border-0">
                  <td className="py-1.5 text-ink-soft">{formatDate(d.date)}</td>
                  <td className="py-1.5 text-ink tabular-nums text-right">
                    {d.weight_kg} kg
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
  });
}

/** Round tick values to clean numbers rather than raw scale bounds. */
function niceTicks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;

  const start = Math.ceil(lo / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= hi; v += step) {
    ticks.push(+v.toFixed(1));
  }
  return ticks;
}
