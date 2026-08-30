"use client";

/** Shared UI primitives. Small, unopinionated, and themed by design tokens. */

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";

export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}

// --------------------------------------------------------------------------
// Button
// --------------------------------------------------------------------------

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-brand-600 text-white hover:bg-brand-700 focus-visible:outline-brand-600 shadow-soft hover:shadow-raised hover:-translate-y-px",
  secondary:
    "bg-card text-ink border border-line hover:bg-raised focus-visible:outline-brand-600",
  ghost:
    "bg-transparent text-ink-soft hover:bg-raised hover:text-ink focus-visible:outline-brand-600",
  danger:
    "bg-clay-600 text-white hover:bg-clay-700 focus-visible:outline-clay-600 shadow-soft hover:shadow-raised hover:-translate-y-px",
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: "px-3 py-1.5 text-sm rounded-lg gap-1.5",
  md: "px-4 py-2.5 text-sm rounded-xl gap-2",
  lg: "px-6 py-3 text-base rounded-xl gap-2",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export function Button({
  variant = "primary",
  size = "md",
  loading = false,
  disabled,
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      disabled={disabled || loading}
      className={cn(
        "inline-flex items-center justify-center font-medium transition-colors",
        "focus-visible:outline-2 focus-visible:outline-offset-2",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
    >
      {loading && <Spinner className="size-4" />}
      {children}
    </button>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn("animate-spin", className)}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
        className="opacity-25"
      />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

// --------------------------------------------------------------------------
// Card
// --------------------------------------------------------------------------

export function Card({
  children,
  className,
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "section" | "article";
}) {
  return (
    <Tag
      className={cn(
        "bg-card border border-line rounded-card shadow-soft",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

export function CardHeader({
  title,
  subtitle,
  action,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 px-5 pt-5 pb-3">
      <div className="min-w-0">
        <h2 className="font-display text-lg font-semibold text-ink">{title}</h2>
        {subtitle && (
          <p className="text-sm text-ink-soft mt-0.5">{subtitle}</p>
        )}
      </div>
      {action}
    </div>
  );
}

// --------------------------------------------------------------------------
// Form fields
// --------------------------------------------------------------------------

export function Field({
  label,
  hint,
  error,
  children,
  htmlFor,
}: {
  label: string;
  hint?: string;
  error?: string;
  children: ReactNode;
  htmlFor?: string;
}) {
  return (
    <div className="space-y-1.5">
      <label
        htmlFor={htmlFor}
        className="block text-sm font-medium text-ink"
      >
        {label}
      </label>
      {children}
      {error ? (
        <p className="text-xs text-clay-600">{error}</p>
      ) : hint ? (
        <p className="text-xs text-ink-muted">{hint}</p>
      ) : null}
    </div>
  );
}

export function Input({
  className,
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "w-full px-3.5 py-2.5 rounded-xl bg-card border border-line text-ink shadow-[inset_0_1px_2px_rgb(43_56_48_/_0.03)]",
        "placeholder:text-ink-muted",
        "focus:outline-2 focus:outline-offset-0 focus:outline-brand-500 focus:border-brand-500",
        "transition-colors",
        className,
      )}
    />
  );
}

/**
 * A selectable card. Used throughout onboarding. Bigger tap targets and room
 * for a description beat a dropdown when the choice actually matters.
 */
export function ChoiceCard({
  selected,
  onSelect,
  emoji,
  title,
  description,
  className,
}: {
  selected: boolean;
  onSelect: () => void;
  emoji?: string;
  title: string;
  description?: string;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "text-left p-4 rounded-2xl border transition-all w-full",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600",
        selected
          ? "border-brand-400 bg-brand-50 dark:bg-brand-950 shadow-soft ring-1 ring-brand-400"
          : "border-line bg-card hover:border-brand-300 hover:bg-raised hover:shadow-soft",
        className,
      )}
    >
      <div className="flex items-start gap-3">
        {emoji && <span className="text-xl leading-none mt-0.5">{emoji}</span>}
        <div className="min-w-0">
          <p
            className={cn(
              "font-medium",
              selected ? "text-brand-700 dark:text-brand-300" : "text-ink",
            )}
          >
            {title}
          </p>
          {description && (
            <p className="text-xs text-ink-soft mt-0.5">{description}</p>
          )}
        </div>
      </div>
    </button>
  );
}

// --------------------------------------------------------------------------
// Feedback
// --------------------------------------------------------------------------

export function Badge({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
        className ?? "bg-raised text-ink-soft",
      )}
    >
      {children}
    </span>
  );
}

export function Alert({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warning" | "error" | "success";
  title?: string;
  children: ReactNode;
}) {
  const tones = {
    info: "bg-brand-50 border-brand-200 text-brand-900 dark:bg-brand-950 dark:text-brand-100",
    success:
      "bg-brand-50 border-brand-300 text-brand-900 dark:bg-brand-950 dark:text-brand-100",
    warning: "bg-spice-100 border-spice-300 text-spice-600",
    error: "bg-clay-100 border-clay-300 text-clay-800",
  };

  return (
    <div className={cn("rounded-2xl border px-4 py-3 text-sm", tones[tone])}>
      {title && <p className="font-semibold mb-0.5">{title}</p>}
      <div>{children}</div>
    </div>
  );
}

/**
 * A labelled progress bar.
 *
 * `value` may exceed `max` (eating over target is a real state), so the fill is
 * clamped for layout while the caption still reports the true number.
 */
export function ProgressBar({
  value,
  max,
  label,
  unit,
  tone = "brand",
}: {
  value: number;
  max: number;
  label: string;
  unit: string;
  tone?: "brand" | "spice" | "clay";
}) {
  const safeMax = max > 0 ? max : 1;
  const pct = Math.min((value / safeMax) * 100, 100);
  const over = value > max;

  const fills = {
    brand: "bg-brand-500",
    spice: "bg-spice-400",
    clay: "bg-clay-500",
  };

  return (
    <div>
      <div className="flex justify-between items-baseline mb-1.5">
        <span className="text-xs font-medium text-ink-soft">{label}</span>
        <span className="text-xs tabular-nums text-ink-muted">
          <span className={cn("font-semibold", over ? "text-clay-600" : "text-ink")}>
            {Math.round(value)}
          </span>
          {" / "}
          {Math.round(max)} {unit}
        </span>
      </div>
      <div className="h-2 bg-raised rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            over ? "bg-clay-500" : fills[tone],
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton rounded-lg", className)} />;
}

export function EmptyState({
  emoji,
  title,
  description,
  action,
}: {
  emoji: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="text-center py-12 px-6">
      <div className="text-4xl mb-3">{emoji}</div>
      <h3 className="font-display text-lg font-semibold text-ink">{title}</h3>
      <p className="text-sm text-ink-soft mt-1 max-w-sm mx-auto">{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
