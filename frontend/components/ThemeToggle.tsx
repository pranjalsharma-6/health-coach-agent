"use client";

import { useCallback, useSyncExternalStore } from "react";

type Theme = "light" | "dark";

const STORAGE_KEY = "kaya-theme";
const CHANGE_EVENT = "kaya-theme-change";

/** The theme lives on <html data-theme>, set before paint by the inline script
 * in `app/layout.tsx`. That attribute is the single source of truth — reading
 * it here rather than keeping a parallel copy in React state means the two can
 * never disagree.
 */
function subscribe(onChange: () => void) {
  window.addEventListener(CHANGE_EVENT, onChange);
  // Another tab switching theme should update this one too.
  window.addEventListener("storage", onChange);
  return () => {
    window.removeEventListener(CHANGE_EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

function getSnapshot(): Theme | null {
  const value = document.documentElement.getAttribute("data-theme");
  return value === "dark" || value === "light" ? value : null;
}

/** The server has no way to know a visitor's system preference, so it renders
 * the neutral state and the client fills it in. `useSyncExternalStore` exists
 * for exactly this and hydrates without a mismatch — reading the DOM in an
 * effect and calling setState would work too, but it is the pattern that
 * causes a flash of the wrong icon.
 */
function getServerSnapshot(): Theme | null {
  return null;
}

export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const toggle = useCallback(() => {
    const next: Theme = getSnapshot() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Private browsing can refuse to store. Losing the preference between
      // visits is not worth breaking the page over.
    }
    window.dispatchEvent(new Event(CHANGE_EVENT));
  }, []);

  const label = theme ? `Switch to ${theme === "dark" ? "light" : "dark"} mode` : "Switch theme";

  return (
    <button
      type="button"
      onClick={toggle}
      // Fixed size so the header doesn't reflow once the theme resolves.
      className="size-9 grid place-items-center rounded-full border border-line bg-card text-ink-soft hover:text-ink hover:bg-raised transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
      aria-label={label}
      title={label}
    >
      <span aria-hidden="true" className="text-sm leading-none">
        {theme === "dark" ? "☀" : theme === "light" ? "☾" : ""}
      </span>
    </button>
  );
}
