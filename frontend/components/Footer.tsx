import { ThemeToggle } from "@/components/ThemeToggle";

/** Site footer.
 *
 * Sits where a copyright line normally would, and says something truer than a
 * copyright line does.
 */
export function Footer() {
  return (
    <footer className="border-t border-line mt-16">
      <div className="max-w-6xl mx-auto px-5 sm:px-8 py-8 flex flex-col sm:flex-row items-center justify-between gap-5">
        <div className="flex items-center gap-2.5 text-sm text-ink-soft order-2 sm:order-1">
          <span>Made with</span>
          <span
            aria-hidden="true"
            className="text-clay-500 text-base leading-none"
          >
            ♥
          </span>
          <span>
            and passion by{" "}
            <a
              href="https://github.com/pranjalsharma-6"
              target="_blank"
              rel="noreferrer"
              className="font-semibold text-ink hover:text-brand-700 dark:hover:text-brand-300 transition-colors underline decoration-brand-300 underline-offset-4 decoration-2"
            >
              Pranjal
            </a>
          </span>
        </div>

        <div className="flex items-center gap-4 order-1 sm:order-2">
          <span className="text-xs text-ink-muted tracking-wide">
            काय · body
          </span>
          <ThemeToggle />
        </div>
      </div>
    </footer>
  );
}
